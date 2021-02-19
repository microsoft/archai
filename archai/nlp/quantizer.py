
from data import SampleIterator, SequentialIterator, evaluate
from modules import AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
from optim import scheduler, get_opt
sys.path.append(Distiller)

class ExplicitQuantize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('max_abs', torch.tensor(1.0))
        self.register_buffer('inv_scale', torch.tensor(1.0))

    def forward(self, x):
        x = x.clamp(min=-self.max_abs, max=self.max_abs)
        return torch.round(x / self.inv_scale) * self.inv_scale

class ExplicitReLUQuantize(ExplicitQuantize):
    def forward(self, x):
        # We can treat this as a fake fused quantized relu
        return super(ExplicitReLUQuantize, self).forward(x.relu())

def Quantize(c):
    if c.run_mode == 'distiller':
        # Distiller will handle this during training and wrap it with range_linear.FakeQuantizationWrapper
        return nn.Identity()
    elif c.run_mode == 'explicit':
        # At evaluation time, we explicitly perform the fake quantization in our code so it's easily verifiable
        return ExplicitQuantize()

def ReLUQuantize(c):
    if c.run_mode == 'distiller':
        return nn.ReLU()
    elif c.run_mode == 'explicit':
        return ExplicitReLUQuantize()

class Decoder(nn.Module):
    def __init__(self, c, layer_i):
        super(Decoder, self).__init__()
        self.layer_i = layer_i
        n_embed = c.n_embed

        self.ln1 = nn.LayerNorm(n_embed)
        self.quant_ln1 = Quantize(c)

        self.qkv = nn.Linear(n_embed, c.n_head * (2 * c.n_k + c.n_v))
        self.quant_qkv = Quantize(c)

        self.pos_emb = nn.Parameter(torch.Tensor(c.n_k, c.n_seq + 1))
        nn.init.normal_(self.pos_emb, 0, 0.02)

        self.quant_attn = Quantize(c)
        self.quant_attnv = Quantize(c)

        self.out = nn.Linear(c.n_head * c.n_v, n_embed, bias=False)
        self.dropout = nn.Dropout(c.dropout)

        self.ln2 = nn.LayerNorm(c.n_embed)
        self.quant_ln2 = Quantize(c)

        self.fc = nn.Sequential(
            nn.Linear(c.n_embed, c.n_inner),
            ReLUQuantize(c),
            nn.Dropout(c.dropout),
            nn.Linear(c.n_inner, c.n_embed),
            Quantize(c),
            nn.Dropout(c.dropout),
        )
        self.c = c

    def forward(self, x, prev=None):
        # x: (n_group * n_seq, n_batch, n_embed)
        # pos_emb: (n_k, n_seq + 1)
        # mask: (2 * n_seq, 2 * n_seq) parallelogram

        c = self.c
        n_s = c.n_seq
        n_g = x.size(0) // n_s
        n_b = x.size(1)
        n_h = c.n_head
        n_bh = n_b * n_h
        n_k = c.n_k
        n_v = c.n_v

        qkv = self.quant_qkv(self.qkv(
            self.quant_ln1(self.ln1(x))
        )).reshape(n_g * n_s, n_b * n_h, 2 * n_k + n_v)
        q, kv = qkv.split([n_k, n_k + n_v], dim=-1)

        q = q.reshape(n_g, n_s, n_b * n_h, n_k)

        padding = prev if prev is not None else torch.zeros((n_s, n_b * n_h, n_k + n_v), dtype=kv.dtype, device=kv.device)
        kv = torch.cat((padding, kv))
        k, v = kv.unfold(0, 2 * n_s, n_s).split([n_k, n_v], dim=2) # (n_g, n_bh, n_kv, 2 * n_s)

        qk = torch.bmm(
            q.transpose(1, 2).reshape(n_g * n_bh, n_s, n_k),
            k.reshape(n_g * n_bh, n_k, 2 * n_s)
        ).reshape(n_g, n_bh, n_s * 2 * n_s).unfold(2, n_s + 1, 2 * n_s + 1) # (n_g, n_bh, n_s, n_s + 1)

        qe = torch.matmul(q, self.pos_emb).transpose(1, 2)

        attn = qk + qe
        attn.mul_(n_k ** -0.5)

        attn = attn.softmax(dim=-1)
        attn = self.quant_attn(attn)

        attn = F.pad(attn, (0, n_s))
        attn = attn.reshape(n_g, n_b * n_h, -1).unfold(2, 2 * n_s, 2 * n_s) # (n_g, n_bh, n_s, 2 * n_s)

        attnv = torch.bmm(
            attn.reshape(n_g * n_bh, n_s, 2 * n_s),
            v.transpose(2, 3).reshape(n_g * n_bh, 2 * n_s, n_v)
        ).reshape(n_g, n_bh, n_s, n_v).transpose(1, 2)
        attnv = self.quant_attnv(attnv)

        attn_out = self.out(attnv.reshape(n_g * n_s, n_b, n_h * n_v)) # (n_g * n_s, n_b, n_embed)
        attn_out = self.dropout(attn_out)

        in_attn = x + attn_out

        out = in_attn + self.fc(self.quant_ln2(self.ln2(in_attn)))
        next = kv[-n_s:].detach()
        return out, next

class Transformer(nn.Module):
    def __init__(self, c):
        super(Transformer, self).__init__()
        self.c = c.setdefault(run_mode='explicit', distributed=False)
        self.embed = AdaptiveEmbedding(c)

        self.dropout1 = nn.Dropout(c.dropout)

        self.layers = nn.ModuleList(Decoder(c, i) for i in range(c.n_layers))

        self.quant = Quantize(c)
        self.dropout2 = nn.Dropout(c.dropout)

        self.loss = ProjectedAdaptiveLogSoftmax(c)

        # tie output embedding weights to input embedding weights
        for layer_embed, layer_loss in zip(self.embed.layers, self.loss.layers):
            layer_loss.weight = layer_embed.weight

    def forward(self, inputs, labels, prevs=None):
        # inputs: (n_group * n_seq, n_batch)
        # labels: (n_group * n_seq, n_batch)
        c = self.c

        n_gs = inputs.size(0)
        n_s = c.n_seq
        if n_gs % n_s != 0: # only the last batch in sequentially iterated data should do this
            padding = torch.zeros((n_s - n_gs % n_s, inputs.size(1)), dtype=inputs.dtype, device=inputs.device)
            inputs = torch.cat((inputs, padding))

        x = self.embed(inputs)
        x = self.dropout1(x)

        prevs = prevs or [None] * c.n_layers
        nexts = []
        for layer, prev in zip(self.layers, prevs):
            x, prev = layer(x, prev=prev)
            nexts.append(prev)

        x = self.quant(x)
        x = self.dropout2(x)
        x = x[:n_gs]

        loss, _ = self.loss(x.reshape(-1, x.size(2)), labels.reshape(-1), keep_order=c.get('keep_order', False))

        loss = loss.reshape(labels.shape)[:n_gs].mean() # this mean is okay because it averages over the sequence (rather than average within a single token)
        return dict(loss=loss, state=nexts)

def train(c):
    import distiller
    net = Transformer(c)

    opt = get_opt(c, net)
    net, opt, step = c.init_model(net, opt=opt, step='max', train=True)

    step_lr = scheduler(c, opt, step)
    data_tr = SampleIterator(c, c.train_batch, split='valid' if c.debug else 'train')
    iter_tr = iter(data_tr)
    data_val = SequentialIterator(c, c.eval_batch, split='valid')
    data_test = SequentialIterator(c, c.eval_batch, split='test')

    print('Before quantization')
    tbl, sparsity = distiller.weights_sparsity_tbl_summary(net, return_total_sparsity=True)
    step_result = pd.Series(evaluate(c, data_val, net)).add_prefix('val_')
    step_result = step_result.append(
        pd.Series(evaluate(c, data_test, net)).add_prefix('test_')
    )
    step_result['sparsity'] = sparsity
    print(step_result)

    compression_scheduler = distiller.config.file_config(net, opt, c.compress)

    print('After initial quantization')
    s = Namespace(net=net, opt=opt, step=step)
    c.on_train_start(s)

    tbl, sparsity = distiller.weights_sparsity_tbl_summary(net, return_total_sparsity=True)
    step_result = pd.Series(evaluate(c, data_val, net)).add_prefix('val_')
    step_result = step_result.append(
        pd.Series(evaluate(c, data_test, net)).add_prefix('test_')
    )
    step_result['sparsity'] = sparsity
    print(step_result)

    npm = []
    for name, param in net.named_parameters():
        if param.dim() in [2, 4] and any(type in name for type in ['weight', 'bias']):
            npm.append((name, param, param.abs() == 0))

    best_val_loss = np.inf
    if s.results is not None and 'val_loss' in s.results.columns:
        best_val_loss = s.results['val_loss'].dropna().max()
    try:
        steps_per_epoch = c.step_eval
        while step < s.step_max:
            epoch = step // steps_per_epoch
            batch = step % steps_per_epoch

            if batch == 0:
                compression_scheduler.on_epoch_begin(epoch)
            compression_scheduler.on_minibatch_begin(epoch, batch, steps_per_epoch)

            step_lr(step)

            x = to_torch(next(iter_tr), c.device).t()

            t_s = time()
            inputs, labels = x[:-1], x[1:]
            preds = net(inputs, labels)
            loss = preds['loss']

            compression_scheduler.before_backward_pass(epoch, batch, steps_per_epoch, loss, False)

            opt.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), c.get('clip_grad', 0.5))

            compression_scheduler.before_parameter_optimization(epoch, batch, steps_per_epoch, opt)
            opt.step()
            for name, param, mask in npm:
                param.data[mask] = 0
            compression_scheduler.on_minibatch_end(epoch, batch, steps_per_epoch)

            if (batch + 1) == steps_per_epoch:
                compression_scheduler.on_epoch_end(epoch)

            time_model = np.round(time() - t_s, 5)

            loss = from_torch(loss)
            perplexity = np.nan if loss > 5 else np.e ** loss
            step_result = pd.Series(dict(
                loss=loss,
                perplexity=perplexity,
                time=time_model,
            )).add_prefix('train_')
            step_result['lr'] = next(iter(opt.param_groups))['lr']

            s.step = step = step + 1
            if step % c.step_eval == 0:
                tbl, sparsity = distiller.weights_sparsity_tbl_summary(net, return_total_sparsity=True)
                step_result = step_result.append(
                    pd.Series(evaluate(c, data_val, net)).add_prefix('val_')
                )
                step_result = step_result.append(
                    pd.Series(evaluate(c, data_test, net)).add_prefix('test_')
                )
                step_result['sparsity'] = sparsity
                s.record_step = step_result['val_loss'] < best_val_loss
                clear_gpu_memory()
            s.step_result = step_result
            c.on_step_end(s)
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        if c.main:
            c.log(err)
        else:
            print(err)
    finally:
        c.on_train_end(s)
    return net, step

def postprocess(c, step):
    print('Postprocessing saved weights to ensure that weights are indeed quantized to %s bits' % c.bits)
    state = net.cpu().state_dict()
    max_int = 2 ** (c.bits - 1) - 1
    min_int = - 2 ** (c.bits - 1)

    import struct

    def float_to_bin(num):
        return bin(struct.unpack('!I', struct.pack('!f', num))[0])[2:].zfill(32)

    def bin_to_float(binary):
        return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

    def get_fp_reduced(x, bits):
        s = float_to_bin(x)
        s = np.array(list(s))
        s[bits:] = '0'
        s = ''.join(s)
        return bin_to_float(s)

    new_state = dict()
    for k, p in net.named_modules():
        if k + '.weight_scale' in state:
            weight = state[k + '.weight']
            weight_scale = state[k + '.weight_scale']
            wq = (weight * weight_scale).round()
            assert (wq - weight * weight_scale).abs().max() < 1e-4
            assert ((min_int <= wq) & (wq <= max_int)).all()
            weight_inv_scale = torch.tensor(
                [get_fp_reduced(x, 32 - c.bits) for x in 1 / weight_scale.reshape(-1)]
            ).reshape_as(weight_scale) # mantissa of weight_inv_scale has at most 32 - c.bits
            new_weight = wq * weight_inv_scale # new weight is exactly representable in 32 bits
            new_state[k + '.weight'] = new_weight
        elif k + '.weight' in state:
            new_state[k + '.weight'] = state[k + '.weight']
        if k + '.bias' in state:
            new_state[k + '.bias'] = state[k + '.bias']
        if k + '.fake_q.scale' in state:
            new_state[k + '.max_abs'] = torch.max(state[k + '.fake_q.tracked_max'].abs(), abs(state[k + '.fake_q.tracked_min'].abs()))
            scale = state[k + '.fake_q.scale']
            inv_scale = get_fp_reduced(1 / scale, 32 - c.bits) # mantissa of inv_scale has at most 32 - c.bits
            new_state[k + '.inv_scale'] = torch.tensor(inv_scale)
        if k + '.pos_emb' in state:
            new_state[k + '.pos_emb'] = state[k + '.pos_emb']
    c.save_state(step, dict(net=new_state, step=step))

if __name__ == '__main__':
    c = Config.from_args().setdefault(run_mode='distiller')
    net, step = train(c)
    postprocess(c, step)
