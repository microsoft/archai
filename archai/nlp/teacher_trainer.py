
from modules import hebbian_weight_update
from optim import scheduler, get_opt
from data import SampleIterator, SequentialIterator, DistillationSampleIterator, evaluate

def train(c):
    c.setdefault(hebbian=False)
    net = eval(c.model)(c)

    emb_params = count_params(net.embed) + count_params(net.loss.projections) + count_params(net.loss.clusters)
    opt = get_opt(c, net)
    net, opt, step = c.init_model(net, opt=opt, step='max', train=True)
    step_lr = scheduler(c, opt, step)

    if c.get('distill'):
        data_tr_distill = DistillationSampleIterator(c, c.train_batch)
        iter_tr_distill = iter(data_tr_distill)
    else:
        data_tr = SampleIterator(c, c.train_batch, split='valid' if c.debug else 'train')
        iter_tr = iter(data_tr)
    data_val = SequentialIterator(c, c.eval_batch, split='valid')

    s = Namespace(net=net, opt=opt, step=step)
    c.on_train_start(s)

    c.log('Embedding has %s parameters' % emb_params)

    if c.hebbian:
        counters = [torch.ones(end - start, dtype=torch.long, device=c.device) for start, end in zip([0] + c.cutoffs, c.cutoffs + [c.n_vocab])]
        temp_counters = [torch.zeros_like(x) for x in counters]

    best_val_loss = np.inf
    if s.results is not None and 'val_loss' in s.results.columns:
        best_val_loss = s.results['val_loss'].dropna().max()
    try:
        while step < s.step_max:
            step_lr(step)
            t_s = time()

            if c.get('distill'):
                hard_labels, soft_labels, soft_probs = next(iter_tr_distill)
                hard_labels = to_torch(hard_labels, c.device).t()

                soft_labels = to_torch(soft_labels, c.device).permute(1, 0, 2)[1:]
                soft_probs = to_torch(soft_probs, c.device).permute(1, 0, 2)[1:]

                inputs, hard_labels = hard_labels[:-1], hard_labels[1:]
                preds = net(inputs=inputs, labels=hard_labels, soft_labels=soft_labels, soft_probs=soft_probs, current_step=step)
            else:
                x = to_torch(next(iter_tr), c.device).t()
                inputs, labels = x[:-1], x[1:]
                preds = net(inputs, labels)
            loss = preds['loss']

            opt.zero_grad()
            if torch.isnan(loss):
                raise RuntimeError('Encountered nan loss during training')
            if c.opt_level == 'O0':
                loss.backward()
            else:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), c.get('clip_grad', 0.5))
            opt.step()

            if c.hebbian:
                hebbian_weight_update(c, net, preds['hiddens'], counters, temp_counters)

            time_model = np.round(time() - t_s, 5)
            loss = from_torch(loss)
            perplexity = np.nan if loss > 5 else np.e ** loss
            step_result = pd.Series(dict(
                loss=loss,
                perplexity=perplexity,
                time=time_model,
            )).add_prefix('train_')
            step_result['lr'] = next(iter(opt.param_groups))['lr']
            if c.get('use_cache'):
                step_result['theta'] = from_torch(preds['theta'])
                step_result['lambda'] = from_torch(preds['lambda'])

            s.step = step = step + 1
            if step % c.step_eval == 0:
                step_result = step_result.append(
                    pd.Series(evaluate(c, data_val, net)).add_prefix('val_')
                )
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

if __name__ == '__main__':
    c = Config.from_args().setdefault(model='model.Transformer')
    evals = [x for x in ['valid', 'test'] if c.get(x)]
    if len(evals):
        net = eval(c.model)(c)
        net, step = c.init_model(net, step=c.get('step', 'max'), train=False)
        print('Model at step', step)

        emb_params = count_params(net.embed) + count_params(net.loss.projections) + count_params(net.loss.clusters)
        print('Model has %s parameters. Embedding has %s parameters' % (count_params(net), emb_params))

        cache_search_path = c.res / ('cache_step%s_n%s.yaml' % (step, c.get('n_cache')))
        if c.get('use_cache_search', True) and cache_search_path.exists():
            for k in 'cache_theta_init', 'cache_lambda_init':
                if c.get(k):
                    c.unvar(k)
            params = cache_search_path.load()
            c.var(**params)
            print('Loaded cache search parameters')
            print(params)

        for split in evals:
            data = SequentialIterator(c, c.eval_batch, split=split)
            print(split, evaluate(c, data, net))
    else:
        train(c)
