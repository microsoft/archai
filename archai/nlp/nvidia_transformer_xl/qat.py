import torch
from torch.nn import functional as F
from torch.quantization import MinMaxObserver

from archai.nlp.nvidia_transformer_xl.nvidia_utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax


class OnnxDynamicObserver(object):
    """Provides a DynamicObserver that is compliant with ONNX-based graphs.

    Note that usually `qint8` is used for symmetric quantization, while
    `quint8` is used for assymetric quantization.

    """

    def __init__(self, dtype):
        """Initializes the class by setting appropriate values for quantization bounds.

        """

        self.dtype = dtype
        self.eps = torch.finfo(torch.float32).eps

        assert dtype in (torch.quint8, torch.qint8)

        if dtype == torch.quint8:
            self.qmin, self.qmax = 0, 255
        else:
            self.qmin, self.qmax = -128, 127

    def __call__(self, x):
        x = x.detach().float()

        self.min_val, self.max_val = x.min().view(-1), x.max().view(-1)

    def calculate_qparams(self):
        if self.dtype == torch.qint8:
            scale = torch.max(self.max_val.clamp(min=0), -self.min_val.clamp(max=0)) / 127

            zero_point = torch.zeros_like(scale).to(torch.int64)

            return scale.clamp(min=self.eps), zero_point
        else:
            scale = (self.max_val - self.min_val) / float(self.qmax - self.qmin)
            scale = scale.clamp(min=self.eps)

            zero_point = self.qmin - torch.round(self.min_val / scale)
            zero_point = zero_point.clamp(min=self.qmin, max=self.qmax).to(torch.int64)

            return scale, zero_point


class FakeDynamicQuant(torch.nn.Module):
    """Inserts a fake dynamic quantizer to allow for a proper scale/zero point calculating
    when performing Quantization Aware Training.

    """
        
    def __init__(self,
                 reduce_range=True,
                 dtype=torch.quint8,
                 bits=8,
                 onnx_compatible=False):
        """Initializes a customizable operator for inserting a fake dynamic quantizer.

        """

        super().__init__()

        self.bits = bits
        self.reduce_range = reduce_range if bits == 8 else False
        self.dtype = dtype
        self.onnx_compatible = onnx_compatible

        assert dtype in (torch.quint8, torch.qint8)

        if dtype == torch.quint8:
            if self.reduce_range:
                self.QMIN, self.QMAX = 0, 2 ** (bits - 1)
            else:
                self.QMIN, self.QMAX = 0, 2 ** bits - 1
        else:
            if self.reduce_range:
                self.QMIN, self.QMAX = -2 ** (bits - 2) ,  2 ** (bits - 2) - 1
            else:
                self.QMIN, self.QMAX = -2 ** (bits - 1) ,  2 ** (bits - 1) - 1

    def forward(self, x):
        if x.dtype == torch.float32:
            if self.bits == 8:
                if self.dtype == torch.quint8:
                    qscheme = torch.per_tensor_affine
                else:
                    qscheme = torch.per_tensor_symmetric
                
                if self.onnx_compatible:
                    observer = OnnxDynamicObserver(dtype=self.dtype)
                else:
                    observer = MinMaxObserver(dtype=self.dtype,
                                              qscheme=qscheme,
                                              reduce_range=self.reduce_range)

                observer(x)
                scale, zero_point = observer.calculate_qparams()
            else:
                min_val, max_val = x.min(), x.max()

                scale_0 = (max_val - min_val) / float(self.QMAX - self.QMIN)

                zero_point_from_min = self.QMIN - min_val / scale_0
                zero_point_from_max = self.QMAX - max_val / scale_0
                zero_point_from_min_error = abs(self.QMIN) - abs(min_val / scale_0)
                zero_point_from_max_error = abs(self.QMAX) - abs(max_val / scale_0)

                if zero_point_from_min_error < zero_point_from_max_error:
                    initial_zero_point = zero_point_from_min
                else:
                    initial_zero_point = zero_point_from_max

                zero_point_0 = initial_zero_point.round()
                scale, zero_point = scale_0, zero_point_0

            x = torch.fake_quantize_per_tensor_affine(x,
                                                      float(scale.item()),
                                                      int(zero_point.item()),
                                                      self.QMIN,
                                                      self.QMAX)

            self._scale, self._zero_pointer = scale, zero_point

        return x


class FakeQuantEmbedding(torch.nn.Embedding):
    """Translates a torch-based Embedding layer into a QAT-ready Embedding.

    """

    def __init__(self, *args, **kwargs):
        """Initializes a fake quantized Embedding layer.

        """

        bits = kwargs.pop('bits') if 'bits' in kwargs else 8
        onnx_compatible = kwargs.pop('onnx_compatible') if 'onnx_compatible' in kwargs else False

        super().__init__(*args, **kwargs)

        self.weight_fake_quant = FakeDynamicQuant(dtype=torch.qint8,
                                                  reduce_range=False,
                                                  bits=bits,
                                                  onnx_compatible=onnx_compatible)

    @property
    def fake_quant_weight(self):
        if self.training:
            return self.weight_fake_quant(self.weight)

        if not hasattr(self, '_quant_weight'):
            self._quant_weight = self.weight_fake_quant(self.weight)

        return self._quant_weight

    def forward(self, x):
        return self.fake_quant_weight[x]

    @classmethod
    def from_float(cls, mod, qconfig=None, **kwargs):
        module = cls(mod.num_embeddings, mod.embedding_dim, **kwargs)

        module.weight = mod.weight
        module.weight.model_parallel = False

        return module

    def to_float(self):
        module = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)

        module.weight.data = self.weight_fake_quant(self.weight.data)
        module.weight.model_parallel = True

        return module


class FakeQuantEmbeddingForOnnx(FakeQuantEmbedding):
    """Allows a QAT-ready Embedding layer to be exported with ONNX.

    """

    def __init__(self, *args, **kwargs):
        """Initializes a fake quantized Embedding layer compatible with ONNX.
        
        """
        
        kwargs['onnx_compatible'] = True

        super().__init__(*args, **kwargs)


class FakeDynamicQuantLinear(torch.nn.Linear):
    """Translates a torch-based Linear layer into a QAT-ready Linear.

    """

    _FLOAT_MODULE = torch.nn.Linear

    def __init__(self,
                 *args,
                 dynamic_weight=True,
                 activation_reduce_range=True,
                 bits=8,
                 onnx_compatible=False,
                 qconfig=None,
                 **kwargs):
        """Initializes a fake quantized Linear layer.
        
        """

        super().__init__(*args, **kwargs)

        self.dynamic_weight = dynamic_weight
        if dynamic_weight:
            self.weight_fake_quant = FakeDynamicQuant(dtype=torch.qint8,
                                                      reduce_range=False,
                                                      bits=bits,
                                                      onnx_compatible=onnx_compatible)

        self.input_pre_process = FakeDynamicQuant(reduce_range=activation_reduce_range,
                                                  bits=bits,
                                                  onnx_compatible=onnx_compatible)

    @property
    def fake_quant_weight(self):
        if self.training:
            return self.weight_fake_quant(self.weight)

        if not hasattr(self, '_quant_weight'):
            self._quant_weight = self.weight_fake_quant(self.weight)

        return self._quant_weight

    def forward(self, x):
        x = self.input_pre_process(x)
        return F.linear(x, self.fake_quant_weight, self.bias)

    @classmethod
    def from_float(cls, mod, qconfig=None, activation_reduce_range=True, **kwargs):
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + cls._FLOAT_MODULE.__name__

        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'
            qconfig = mod.qconfig

        qat_linear = cls(mod.in_features,
                         mod.out_features,
                         bias=mod.bias is not None,
                         activation_reduce_range=activation_reduce_range,
                         qconfig=qconfig,
                         **kwargs)

        qat_linear.weight = mod.weight
        qat_linear.bias = mod.bias

        return qat_linear

    def to_float(self):
        weight = self.weight_fake_quant(self.weight)

        float_linear = torch.nn.Linear(self.in_features,
                                       self.out_features,
                                       bias=self.bias is not None)

        float_linear.weight = torch.nn.Parameter(weight)
        float_linear.bias = self.bias

        return float_linear


class FakeDynamicQuantLinearForOnnx(FakeDynamicQuantLinear):
    """Allows a QAT-ready Linear layer to be exported with ONNX.

    """

    def __init__(self, *args,  **kwargs):
        """Initializes a fake quantized Linear layer compatible with ONNX.
        
        """

        kwargs['activation_reduce_range'] = False
        kwargs['onnx_compatible'] = True

        super().__init__(*args, **kwargs)

class FakeDynamicQuantConv1d(torch.nn.Conv1d):
    """Translates a torch-based Conv1D layer into a QAT-ready Linear.

    """

    _FLOAT_MODULE = torch.nn.Conv1d

    def __init__(self,
                 *args,
                 dynamic_weight=True,
                 activation_reduce_range=True,
                 bits=8,
                 onnx_compatible=False,
                 qconfig=None,
                 **kwargs):
        """Initializes a fake quantized Conv1D layer.
        
        """

        super().__init__(*args, **kwargs)

        self.dynamic_weight = dynamic_weight
        if dynamic_weight:
            self.weight_fake_quant = FakeDynamicQuant(dtype=torch.qint8,
                                                      reduce_range=False,
                                                      bits=bits,
                                                      onnx_compatible=onnx_compatible)

        self.input_pre_process = FakeDynamicQuant(reduce_range=activation_reduce_range,
                                                  bits=bits,
                                                  onnx_compatible=onnx_compatible)

    @property
    def fake_quant_weight(self):
        if self.training:
            return self.weight_fake_quant(self.weight)

        if not hasattr(self, '_quant_weight'):
            self._quant_weight = self.weight_fake_quant(self.weight)

        return self._quant_weight

    def forward(self, x):
        x = self.input_pre_process(x)
        return self._conv_forward(x, self.fake_quant_weight, self.bias)

    @classmethod
    def from_float(cls, mod, qconfig=None, activation_reduce_range=True, **kwargs):
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + cls._FLOAT_MODULE.__name__

        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'
            qconfig = mod.qconfig

        qat_conv1d = cls(in_channels=mod.in_channels,
                         out_channels=mod.out_channels,
                         kernel_size=mod.kernel_size,
                         stride=mod.stride,
                         padding=mod.padding,
                         dilation=mod.dilation,
                         groups=mod.groups,
                         padding_mode=mod.padding_mode,
                         bias=mod.bias is not None,
                         activation_reduce_range=activation_reduce_range,
                         qconfig=qconfig,
                         **kwargs)

        qat_conv1d.weight = mod.weight
        qat_conv1d.bias = mod.bias

        return qat_conv1d

    def to_float(self):
        weight = self.weight_fake_quant(self.weight)

        float_conv1d = torch.nn.Conv1d(in_channels=self.in_channels,
                                       out_channels=self.out_channels,
                                       kernel_size=self.kernel_size,
                                       stride=self.stride,
                                       padding=self.padding,
                                       dilation=self.dilation,
                                       groups=self.groups,
                                       padding_mode=self.padding_mode,
                                       bias=self.bias is not None)

        float_conv1d.weight = torch.nn.Parameter(weight)
        float_conv1d.bias = self.bias

        return float_conv1d

class FakeDynamicQuantConv1dForOnnx(FakeDynamicQuantConv1d):
    """Allows a QAT-ready Linear layer to be exported with ONNX.

    """

    def __init__(self, *args,  **kwargs):
        """Initializes a fake quantized Linear layer compatible with ONNX.
        
        """

        kwargs['activation_reduce_range'] = False
        kwargs['onnx_compatible'] = True

        super().__init__(*args, **kwargs)

# Maps between standard and ONNX modules
DYNAMIC_QAT_MODULE_MAPPING = {
    torch.nn.Embedding: FakeQuantEmbedding,
    torch.nn.Linear: FakeDynamicQuantLinear,
    torch.nn.Conv1d: FakeDynamicQuantConv1d
}
DYNAMIC_QAT_MODULE_MAPPING_FOR_ONNX = {
    torch.nn.Embedding: FakeQuantEmbeddingForOnnx,
    torch.nn.Linear: FakeDynamicQuantLinearForOnnx,
    torch.nn.Conv1d: FakeDynamicQuantConv1dForOnnx
}

# Adds placeholder for changing `_compute_logit`
COMPUTE_LOGIT = ProjectedAdaptiveLogSoftmax._compute_logit


def dynamic_qat_compute_logit(self, hidden, weight, bias, proj):
    """Translates `_compute_logit` from Adaptive Softmax to a QAT-ready version.

    """

    # if no projection then simply multiply hidden values with wights
    # else apply projection to hidden and then multiply with weight matrix
    if proj is None:
        fake_quant_hidden = self.hidden_fake_quant(hidden)
        fake_quant_weight = self.weight_fake_quant(weight)
        logit = F.linear(fake_quant_hidden, fake_quant_weight, bias=bias)
    else:
        # below is equivalent to:
        # proj_hid = nn.functional.linear(hidden, proj.t().contiguous())
        # logit = nn.functional.linear(proj_hid, weight, bias=bias)
        logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
        if bias is not None:
            logit = logit + bias
    return logit


def qat_to_float_modules(model):
    """Changes QAT-ready modules to float-based modules.

    """

    for name in list(model._modules):
        module = model._modules[name]            

        if hasattr(module, 'to_float'):
            model._modules[name] = module.to_float()
        else:
            qat_to_float_modules(module)

    ProjectedAdaptiveLogSoftmax._compute_logit = COMPUTE_LOGIT

    return model


def float_to_qat_modules(model,
                         module_mapping=DYNAMIC_QAT_MODULE_MAPPING,
                         qconfig=None,
                         **kwargs):
    """Changes float-based modules to QAT-ready modules.

    """

    for name in list(model._modules):
        module = model._modules[name]

        if type(module) in module_mapping:
            if not hasattr(module, 'qconfig'):
                module.qconfig = qconfig

            model._modules[name] = module_mapping[type(module)].from_float(module, qconfig, **kwargs)
        else:
            float_to_qat_modules(module,
                                 module_mapping=module_mapping,
                                 qconfig=qconfig,
                                 **kwargs)

    ProjectedAdaptiveLogSoftmax.hidden_fake_quant = FakeDynamicQuant(dtype=torch.qint8,
                                                                     onnx_compatible=True)
    ProjectedAdaptiveLogSoftmax.weight_fake_quant = FakeDynamicQuant(dtype=torch.qint8,
                                                                     onnx_compatible=True)
    ProjectedAdaptiveLogSoftmax._compute_logit = dynamic_qat_compute_logit

    return model


def prepare_with_qat(model,
                     onnx_compatible=False,
                     backend='qnnpack',
                     **kwargs):
    """Prepares a float-based model and inserts QAT-based modules and configurations.

    """

    # Gathers the `qconfig` and appropriate modules mappings
    qconfig = torch.quantization.get_default_qat_qconfig(backend)
    mappings = DYNAMIC_QAT_MODULE_MAPPING_FOR_ONNX if onnx_compatible else DYNAMIC_QAT_MODULE_MAPPING

    # Ensures that the model is QAT-ready
    float_to_qat_modules(model,
                         module_mapping=mappings,
                         qconfig=qconfig,
                         **kwargs)

    return model
