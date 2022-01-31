import copy

from archai.nlp.models.model_base import ArchaiModel
from archai.nlp.compression.quantization.qat import prepare_with_qat

class MixedQATModel(ArchaiModel):
    """A wrapper for performing mixed QAT training. A new fake quantized model
    is created and share its weights with the regular model. The training loss
    is calculated as follows: loss = qat_loss * qat_weight + regular_loss * (1 - qat_weight).
    The inference loss is the same as qat_loss.
    """

    def __init__(self, model, qat_weight=0.2) -> None:
        super(MixedQATModel, self).__init__()

        if qat_weight < 0.0 or qat_weight > 1.0:
            raise ValueError(f'QAT weight should be between 0 and 1 ({qat_weight})')

        self.qat_weight = qat_weight
        self.regular_weight = 1.0 - qat_weight
        self.model = model
        self.qat_model = copy.deepcopy(model)

        # Share all the parameters
        for module1, module2 in zip(self.model.modules(), self.qat_model.modules()):
            if hasattr(module2, 'weight'):
                module2.weight = module1.weight
            if hasattr(module2, 'bias'):
                module2.bias = module1.bias

        # Add fake quantization
        self.qat_model = prepare_with_qat(self.qat_model, onnx_compatible=True)

        # Make sure all the parameters are shared
        for param1, param2 in zip(self.model.parameters(), self.qat_model.parameters()):
            assert param2 is param1, 'Mixed QAT parameters are not fully shared'

    def forward(self, *args, **kwargs):

        out = self.model(*args, **kwargs)
        qat_out = self.qat_model(*args, **kwargs)

        # If we are not training we only return the qat loss
        if self.training:
            return ((out[0]*self.regular_weight + qat_out[0]*self.qat_weight), out[1], out[2], out[3])
        else:
            return qat_out

    def reset_length(self, *args, **kwargs):
        return self.model.reset_length(*args, **kwargs)

    def get_non_emb_params(self):
        return self.model.get_non_emb_params()
