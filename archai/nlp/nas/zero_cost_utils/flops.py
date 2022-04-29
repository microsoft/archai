import torch

from archai.nlp.models.mem_transformer.model_mem_transformer import PositionwiseFF, RelLearnableMultiHeadAttn, RelMultiHeadAttn, MultiHeadAttn, RelPartialLearnableMultiHeadAttn
from archai.nlp.models.model_utils.adaptive_embedding import AdaptiveEmbedding
from archai.nlp.models.model_utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax


def get_list_of_layers(module, layerType=None):
    # returns a list of layers (optionally with a certain type) that have trainable parameters
    
    submodules = list(module.children())
    list_of_layers = []
    
    if layerType is not None:
        for lt in layerType:
            if isinstance(module, lt):
                return module
    else:
        if len(submodules)==0 and len(list(module.parameters()))>0:
            return module
    
    for m in submodules:
            try:
                list_of_layers.extend(get_list_of_layers(m, layerType))
            except TypeError:
                list_of_layers.append(get_list_of_layers(m, layerType))
    
    return list_of_layers

def get_in_out_shape(self, input, output):
    self.input_size = torch.tensor(input[0].size())
    self.output_size = torch.tensor(output.size())

def get_layer_flops(l):
    if isinstance(l, AdaptiveEmbedding):
        if len(l.emb_projs) > 0:
            return torch.prod(l.output_size) * l.emb_projs[0].size(-1)
        else:
            return torch.tensor([0])

    elif isinstance(l, PositionwiseFF):
        return (torch.prod(l.input_size) + torch.prod(l.output_size)) * l.d_inner

    elif isinstance(l, RelPartialLearnableMultiHeadAttn):
        return l.flops

    elif isinstance(l, ProjectedAdaptiveLogSoftmax):
        return l.flops

    else:
        raise NotImplemented

def get_model_flops(model, inp, tgt):
  layers_with_flops = get_list_of_layers(model, layerType=[AdaptiveEmbedding, PositionwiseFF, MultiHeadAttn, RelMultiHeadAttn, \
                                                            RelPartialLearnableMultiHeadAttn, RelLearnableMultiHeadAttn, ProjectedAdaptiveLogSoftmax])
  
  # register forward hooks to record input and output sizes
  hooks = []
  for l in layers_with_flops:
    h = l.register_forward_hook(get_in_out_shape)
    hooks.append(h)
  
  model(inp, tgt, mems=None)

  flops = {}
  for l in layers_with_flops:
    f = get_layer_flops(l)
    
    if isinstance(l, AdaptiveEmbedding):
      key = 'AdaEmb'
    elif isinstance(l, PositionwiseFF):
      key = 'FFN'
    elif isinstance(l, RelPartialLearnableMultiHeadAttn):
      key = 'Attn'
    elif isinstance(l, ProjectedAdaptiveLogSoftmax):
      key = 'Sftmax'
    else:
      raise NotImplemented
      
    if key in flops.keys():
      flops[key] += f.item()
    else:
      flops[key] = f.item()
    
  return flops