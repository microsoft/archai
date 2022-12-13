Implementing DARTS
==================

DARTS is a differentiable architecture search algorithm. In this document, we will describe how to implement DARTS in Archai as an example. Note that this algorithm is already implemented and can be found in :github:`algos/darts`.

To implement DARTS, we will first create the ``MixedOp`` operation that combines all available operations along with their architecture weights. We will then use the ``MixedOp`` to create a super network with all possible edges. To train this super network, we will override the ``ArchTrainer`` class and use a bi-level optimizer.

MixedOp
-------

The ``MixedOp`` class is responsible for combining the output of all available operations, weighted by their architecture parameters. The ``forward()`` method of the ``MixedOp`` class sums the output of the operations, using the softmax of the architecture parameters as weights:

.. code-block:: python

    class MixedOp(Op):
        ...
        def forward(self, x):
            asm = F.softmax(self._alphas[0], dim=0)
            return sum(w * op(x) for w, op in zip(asm, self._ops))

Note that the ``MixedOp`` class creates one architecture parameter for each primitive operation, and keeps them encapsulated within the ``Op`` class. The ``nn.Module`` class only has a ``parameters()`` method to retrieve learned weights, and does not differentiate between architecture weights and regular weights. The ``Op`` class, however, allows us to separate these two types of parameters.

Another important method in the ``MixedOp`` class is the ``finalize()`` method, which chooses the top primitives by architecture weight and returns them:

.. code-block:: python

    class MixedOp(Op):
        ...
        def finalize(self) -> Tuple[OpDesc, Optional[float]]:
            # return finalized op description and its weight
            with torch.no_grad():
                # select except 'none' op
                val, i = torch.topk(self._alphas[0][:-1], 1)
                desc, _ = self._ops[i].finalize()
                return desc, float(val.item())

ModelDescBuilder
----------------

The ``ModelDescBuilder`` class is responsible for building the super network that the searcher will use. The ``ModelDescBuilder`` builds the model description in parts: first, the model stems, then each cell, and finally, the pooling and logits layers. Within each cell, we first build the cell stems, then the nodes and their edges, and finally, a layer called the "post op" that produces the output. Each of these steps is implemented in its own method, allowing for easy customization.

For DARTS, we just need to build nodes with ``MixedOp`` as edges. To do this, we override the ``build_nodes()`` method in the ``DartsModelDescBuilder`` class:

.. code-block:: python

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

The parameters of this method indicate the expected input and output shapes for each node, as well as the cell type indicating whether it is a regular or reduction cell. The core of the method simply creates the ``NodeDesc`` instances to represent each node.

Trainer
-------

To perform a search, DARTS uses a bi-level optimization algorithm. To implement this, it is necessary to separate the regular weights from the architecture weights. The architecture weights are then trained using the bi-level optimizer. This can be easily done by taking advantage of the "hooks" provided by the trainer. These include the ``pre_fit()`` and ``post_fit()`` hooks, which are executed before and after the code for the ``fit()`` method. In the ``pre_fit()`` hook, the ``BilevelOptimizer`` class is initialized:

.. code-block:: python

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

In the ``pre_step()`` hook, a step is run on the ``BilevelOptimizer``:

.. code-block:: python

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

Putting All Together
--------------------

Once the custom ``Trainer`` and ``ModelDescBuilder`` classes have been created for DARTS, they must be specified to Archai through a class derived from ``ExperimentRunner.`` This is accomplished by overriding the ``model_desc_builder`` and ``trainer_class`` methods to specify the custom classes.

.. code-block:: python

    class DartsExperimentRunner(ExperimentRunner):
        def model_desc_builder(self)->DartsModelDescBuilder:
            return DartsModelDescBuilder()

        def trainer_class(self)->TArchTrainer:
            return BilevelArchTrainer

Finally, the algorithm name and ``DartsExperimentRunner`` must be added to ``main.py`` so that they are utilized when ``darts`` is specified in the ``--algos`` switch.

.. code-block:: python

    def main():
        ...
        runner_types:Dict[str, Type[ExperimentRunner]] = {
            'darts': DartsExperimentRunner,
            ...
        }
