Implementing DARTS
==================

We will now do quick walkthrough on how we can implement DARTS in Archai as an example. Note that this algorithm is already implemented so you can see the [final code](https://github.com/microsoft/archai/tree/master/archai/algos/darts).

At high level, we will first create the the op that combines all ops along with their architecture weights. We will call this `MixedOp`. We will then use the `MixedOp` to create super network with all possible edges. To train this super network, we will override `ArchTrainer` and use bi-level optimizer. After the model is trained, we will use `Finalizers` class to generate the final model description. Finally, we will just use default `Evaluater` to evaluate the model.

Implementing MixedOp

The main idea is to simply create all 7 primitives DARTS needs and override the `forward` method as usual to sum the output of primitives weighted by architecture parameters.

```python
class MixedOp(Op):
    ...
    def forward(self, x):
        asm = F.softmax(self._alphas[0], dim=0)
        return sum(w * op(x) for w, op in zip(asm, self._ops))
```

Notice that we create one architecture parameter for each primitive and they stay encapsulated within that instance of `Op` class. The `nn.Module` only has `parameters()` method to retrieve learned weights and does not differentiate between architecture weights vs. the regular weights. The `Op` class however allows us to separate these two types of parameters.

Another method to focus on is `finalize` which chooses top primitives by architecture weight and returns  it.

```python
class MixedOp(Op):
    ...
    def finalize(self) -> Tuple[OpDesc, Optional[float]]:
        # return finalized op description and its weight
        with torch.no_grad():
            # select except 'none' op
            val, i = torch.topk(self._alphas[0][:-1], 1)
            desc, _ = self._ops[i].finalize()
            return desc, float(val.item())
```

[View full code](https://github.com/microsoft/archai/blob/master/archai/algos/darts/mixed_op.py)

Implementing the ModelDescBuilder

The job of `ModelDescBuilder` is to build the super network that searcher can use. The `ModelDescBuilder` builds the model description in parts: first model stems, then each cell and finally pooling and logits layers. Within each cell we first build cell stems, then nodes and their edges and finally a layer we will call "post op" that produces the output. Each of these steps are implemented in their own methods so you can override any portion of model building and customize according to your needs.

For DARTS, we just need to build nodes with `MixedOp` as edges. For this we override the `build_nodes` method.

```python
class DartsModelDescBuilder(ModelDescBuilder):
    ...
    def build_nodes(self, stem_shapes:TensorShapes,
                    conf_cell:Config,
                    cell_index:int, cell_type:CellType,
                    node_count:int,
                    in_shape:TensorShape, out_shape:TensorShape) \
                        ->Tuple[TensorShapes, List[NodeDesc]]:

        # is this cell reduction
        reduction = (cell_type==CellType.Reduction)

        # create nodes list
        nodes:List[NodeDesc] =  []

        # input and output channels for each node
        conv_params = ConvMacroParams(in_shape[0], out_shape[0])

        # for each noce we will create NodeDesc object
        for i in range(node_count):
            # for each node we have incoming edges
            edges=[]
            # each node connects back to all previous nodes and s0 and s1 states
            for j in range(i+2):
                # create MixedOp for each edge
                op_desc = OpDesc('mixed_op',
                                    params={
                                        # in/out channels for the edhe
                                        'conv': conv_params,
                                        # if reduction cell than use stride=2
                                        # for the stems
                                        'stride': 2 if reduction and j < 2 else 1
                                    },
                                    # MixedOp only takes one input
                                    in_len=1)
                # Edge description specifies op and where its input(s) comes from
                edge = EdgeDesc(op_desc, input_ids=[j])
                edges.append(edge)

            # add the node in our collection
            nodes.append(NodeDesc(edges=edges, conv_params=conv_params))

        # we need to return output shapes for each node which is same as input
        out_shapes = [copy.deepcopy(out_shape) for _  in range(node_count)]

        return out_shapes, nodes
```

Notice that the parameters of this method tell us the expected input and output shape for each node, the cell type indicating whether it's a regular or reduction cell and so on. The core of the method simply creates the `NodeDesc` instances to represent each node.

[View full code](https://github.com/microsoft/archai/blob/master/archai/algos/darts/darts_model_desc_builder.py)

Implementing the Trainer

To perform a search, DARTS uses bi-level optimization algorithm. To implement this, we need to separate regular weights from architecture weights. We then train the architecture weights using the bi-level optimizer. This can be done easily by taking advantage of *hooks* that the trainer provides. These include `pre_fit` and `post_fit` hooks that get executed before and after the code for the `fit` method. So, in `pre_fit` we can initialize our `BilevelOptimizer` class.

```python
class BilevelArchTrainer(ArchTrainer):
    ...
    def pre_fit(self, data_loaders:data.DataLoaders)->None:
        super().pre_fit(data_loaders)

        # get config params for bi-level optimizer
        w_momentum = self._conf_w_optim['momentum']
        w_decay = self._conf_w_optim['decay']
        lossfn = ml_utils.get_lossfn(self._conf_w_lossfn).to(self.get_device())

        # create bi-level optimizer
        self._bilevel_optim = BilevelOptimizer(self._conf_alpha_optim,
                                                w_momentum,
                                                w_decay, self.model, lossfn)
```

 Then we use `pre_step` hook to run a step on `BilevelOptimizer`.

 ```python
class BilevelArchTrainer(ArchTrainer):
    ...
    def pre_step(self, x: Tensor, y: Tensor) -> None:
        super().pre_step(x, y)

        # get the validation dataset for bi-level optimizer
        x_val, y_val = next(self._valid_iter)

        # get regular optimizer
        optimizer = super().get_optimizer()

        # update alphas
        self._bilevel_optim.step(x, y, x_val, y_val, optimizer)
 ```

[View full code](https://github.com/microsoft/archai/blob/master/archai/algos/darts/bilevel_arch_trainer.py)

Putting It All Togather

Now that we have our own `Trainer` and `ModelDescBuilder` for DARTS, we need to tell Archai about them. This is done through a class derived from `ExperimentRunner`. We override `model_desc_builder()` and `trainer_class()` to specify our custom classes.

```python
class DartsExperimentRunner(ExperimentRunner):
    def model_desc_builder(self)->DartsModelDescBuilder:
        return DartsModelDescBuilder()

    def trainer_class(self)->TArchTrainer:
        return BilevelArchTrainer
```

[View full code](https://github.com/microsoft/archai/blob/master/archai/algos/darts/darts_exp_runner.py)

Finally, add our algorithm name and `DartsExperimentRunner` in `main.py` so it gets used when `darts` is specified in `--algos` switch.

```python
def main():
    ...
    runner_types:Dict[str, Type[ExperimentRunner]] = {
        'darts': DartsExperimentRunner,
        ...
    }
```

[View full code](https://github.com/microsoft/archai/blob/master/scripts/main.py)
