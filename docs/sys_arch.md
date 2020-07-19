# System Architecture

## TODO
- Fix yaml indent: https://github.com/Microsoft/vscode/issues/42771
- yaml anchors tests
- remove cutout from utils
- remove .weights()-  doesn't include stems, projection ops etc
- implement reduced datasets using previous code
- log batch size
- Toy pareto, test philly job
- detect multi instances of the script
- dump command line in a file
- darts non-pareto ref run
- node num pareto as array in yaml
- accept filepath for found model instead of assuming in expdir
- eval trains all models in search dir
- distributed search run, aggregation script, generate distributed eval run yaml
- convex hull code
- test darts and random search pareto
- debug slow model forward and backward
- checkpoint
- measure multiple run variances
- construct dawnnet
- construct better search space for fast forward and backward
- imagenet transfer measurement
- GPU utilization is too low


## Model Compiler Options

- Macro builder will add auxtowers in eval
- DagEdge will apply droppath in eval
- BatchNorms will be affine in eval
- 0 cell models are valid
- if cell is present, it must have at lease out_states
- Nodes may not have any edge

## Search

### Algorithm

For Darts and Random search:

```
input: conf_macro, cell_builder
output: final_desc

macro_desc = build_macro(conf_macro)
model_desc = build_desc(macro_desc, cell_builder)
model = build_model(model_desc)
train(model)
final_desc = finalize(model)
```

For PetriDish, we need to add n iteration

```
input: conf_macro, cell_builder, n_search_iter
output: final_desc

macro_desc = build_macro(conf_macro)
for i = 1 to n_search_iter:
    if pre_train_epochs > 0:
        if all nodes non-empty:
            model = build_model(model_desc, restore_state=True)
            train(mode, pre_train_epochsl)
            macro_desc = finalize(model. include_state=True)
        elif all nodes empty:
            pass because no point in training empty model
        else
            raise exception

    # we have P cells, Q nodes each with 0 edges on i=1 at this point
    # for i > 1, we have P cells, i-1 nodes at this point
    # Petridish micro builder removes 0 edges nodes after i
    # if number of nodes < i, Petridish macro adds nodes
    # assert 0 edges for all nodes for i-1
    # Petridish micro builder adds Petridish op at i
    model_desc = build_desc(macro_desc, cell_builder(i))
    # we have P cells, i node(s) each
    model = build_model(model_desc, restore_state=True)
    arch_train(model)
    macro_desc = final_desc = finalize(model. include_state=True)
    # make sure FinalPetridishOp can+will run in search mode
    # we end with i nodes in each cell for Petridish at this point
```

### Checkpointing search

Loop1: search iterations
    Loop2: pre-training
    Loop3: arch-training

Each loop has state and current index.

Cases:
    termination before Loop1
    termination before Loop2
    termination during Loop2
    termination after Loop2
    termination before Loop3
    termination during Loop3
    termination after Loop3
    termination after Loop1

Idea:
    Each node maintains its unique key in checkpoint
    Each node updates+saves checkpoint *just after* its iteration
        Checkpoint can be saved any time
    When node gets checkpoint, if it finds own key
        it restores state, iteration and continues that iteration

## Logging

We want logs to be machine readable. To that end we can think of log as dictionary. One can insert new key, value pair in this dictionary but we should allow to overwrite existing values unless value themselves are container type in which case, the log value is appended in that container. Entire log is container itself of type dictionary. ANothe container is array.

log is class derived from ordered dict. Insert values as usual. key can be option in which case internal counter may be used. It has one additional method child(key) which returns log object inserted at the key.


```
logger.add(path, val, severity=info)

path is string or tuple. If tuple then it should consist of ordered dictionary keys.

logger.add('cuda_devices', 5)
logger.add({'cuda_devices': 5, 'cuda_ver':4})
logger.add(('epochs', 5), {acc=0.9, time=4.5})
logger.add(('epochs', 5), {acc=0.9, time=4.5})

logger.begin_sec('epochs')
    logger.begin.sec(epoch_i)

        logger.add(key1, val1)
        logger.add({...})


    logger.end_Sec()
longer.end_sec()

```


## Cells and Nodes
Darts Model
    ConvBN
    Cell
        ReLUSepConv/2BN if reduction else ReLUSepConvBN
        sum for each node
        concate channels for all nodes
    AdaptiveAvgPool
    Linear


## Petridish

### Cell Search

#### Constraints

- Every cell of same type must have same number of nodes and ops
- Channels for each cell output remain same iteration over iteration
    - Otherwise, if cell A has different channels then next cells must be
        rebuilt from scratch (i.e. cannot warm start from previous iteration)
    - This implies we must use concate nodes + proj at cell o/p as well
        - this is because we will change number of nodes
        - other option is to use sum of all node outputs
- We can insert new regular cell within each cut but we cannot insert
    new reduction cell because that would change the number of channels
- model starts and ends with regular cells

#### Algo

reduction_cells = 2 or 3
max_reg_cells = k st k >= 1

For given model:
    - Fork 1: add new regular cell in each cur with same number of nodes
    - Fork 2: add new node in all regular cells
    - Fork 3: add new node in all reduction cells

ParitoWorker(TrainerWorker, SearchWorker):
    graph: [(id, model, is_trained, val_acc, flops, parent_id)]
    Take k1 promising trained model
        add a cell
        put back in graph
    Take k2 promising untrained model
        train, put back in graph
    Take k3 promising trained models
        search, put back in graph

## Pareto eval strategy
In new scheme of things, we generate pareto by,

for reductions=r1 to r2
  Generate macro without any warm start
  for depth=d1 to d2
    modify macro to increase depth as needed
    seed model
    for iter=0 to n-1
        call microbuilder

Microbuilder typically only changes cell desc. The seeding process may update
nodes, for ex, add new node+edges or delete some. After seeding process, we train
model and then we call
macrobuilder n times. Each time it may update nodes in any way it likes.
We perform arch search on this, train model again on finalized desc and record its
accuracy.

This process creates many finalized model descs. Current eval howeevr can consume
only one finalized model so we keep last one for current eval.

## Perf observations

* train epoch on cifar10 is 16.8s on Titan Xp, batch=128, resnet34 (21M params).
  This translates to 41ms/step. Out of this, 12.4ms is consumed in forward pass and same in backward pass.
  Test pass takes 2s.
  Same numbers were found on multiple simultaneous jobs on p-infra with 1080ti
  except that forward and backward passes were reported to be 5ms consistently instead of 12ms
  even though overall numbers remain almost exactly the same.
* For darts default search run (batch=64, 8 cells, 4 nodes) in archai on P100
  search:
    epoch time  step_time   run_time
    1647s        4.0s         22.88h
  eval:
    epoch time  step_time   run_time
    255s        0.48s         42.56h
* For darts default search run (batch=64, 8 cells, 4 nodes) in archai on Titan Xp
  search:
    epoch time  step_time   run_time
    1527s       3.4s         21.47h
  eval:
    epoch time  step_time   run_time
    255-2675-?s        0.5s         ?

, one epoch costs 1623s for train and val each. step_time for train was 3.9s.
  Same thing on P100, 1626-1655s/epoch for train as well as val, step_time=4s.
* On 1080Ti, we get 1527-1564s/epoch for training for train and val, step_time=3.2-3.8s.
* For darts default eval run in archai on 1080Ti: 254-281s/epoch initially, step_time=0.5s.
  This grows to 903s/epoch @ epoch 149 2675s/epoch @ epoch 384 while train step_time remains 0.5s.
  Search started at 2020-03-04T07:55, search ended: 2020-03-05T05:23.
  Eval started at 2020-03-05T05:23, eval ended: 2020-03-10T23:10 @ epoch 384.


## Accuracy observations

For darts.
* search best val = 0.8812 @ epoch 42

## Checkpointing
We use following generic model of a program: A program maintains a global state G.
A subset of G is seeded with initial values I. The program performs some computation
and modifies G. If program is stochastic, final G will be stochastic for same I.
The program may consist of child programs cp_i, each of which obtains a subset of G
to operate on.

To enable checkpointing we first must make i in cp_i make part of G as G.cp_i which would
be initialized with 0. Note that this means program must be able to jump to G.cp_i
child program at the start. This means that each child program must have interface
that takes exactly one input G that is mutable. Thus, instead of series of statements
child programs may be represented as list.

We also need to make I as immutable part of G as G.I. The checkpoint
will be referred to as C and it must contain entire G as C.G.
When program is run it is now supplied with I and C. The C may be empty
if its first run or it was interrupted before and contains G at that point in C.G.
At start, the program will continue as normal if C is empty. If C is not empty
then the program must first assert that I supplied to it is same as C.G.I. If it is not
then the program should exit with error other wise program should set its G from C.G,
read G.cp_i and jump to that child program.

What happens when we have hierarchy of child programs? What if level 3 child gets
interrupted? Can we restore to that point? Who should be saving the checkpoint and
at what time is it allowed to save the checkpoint?

First, lets assume that each child i reserves subset of G, G_i = G.cp[i] as its own state
space. Within G_i, child may maintain G_i.I which is its own immutable input and
G_i.cp_i which is its pointer to its own child. The child may modify any non-mutable
part of G as its output.

To enable hierarchical checkpointing, each child must also decompose itself into
its own child just like parent and maintain protocol that its own child consumes
exactly one parameter G.

In this design, checkpoint can be saved only after checkpointable child has been
executed. The parent should increment cp_i and save it.

What if child c_i is not checkpointable but child c_i+1 is checkpointable?
In this case, c_i will do recomputation and make changes to G that could be
stochastically different for same I. If any next checkpointable child access
this G and compare with its checkpoint to find it different and error out.
In other words, this condition produce significant silent or non-silent errors
and confusing behavior. To circumwent this we must ensure if c_i+1 is checkpointable
then every c_i is also. To do this, we need to make cp as tree. At any time,
we should be able to tell where we are in this tree. When a checkpoint is saved,
we check if previous nodes in tree were also saved. If not then we raise error.

As can be seen, hierarchical checkpointing is complex. Also, we need to save G for
all children which may become very large and save of checkpoint can become expensive.
 To simplify things, we may
simply prohibit hierarchical checkpoint. In other words, we only have top level
parent and only that parent is aware of checkpointing and saving of checkpoints.
In this model, parent need to save only its own G that it maintains between child to
child calls. The big downside of this is that if any of the child is long running
than any other then we can't checkpoint within that specific child.

## Yaml design
Copy node value using '_copy :/path/to/node'.
    - target will always be scaler
    - source could be dict or scaler
    - recursive replacements
    - this will replace above string path with target value
Insert node childs using _copy: /path/to/node
    - content of source node is copied
    - rest of the child overrides

## Pytorch parameter naming convention

- Parameter object is same as Tensor
- Within each module, parameter gets named by variable name it is assigned to
- Parameter is only added in .parameters() or .named_parameters() if its instance wasn't already added
- Optimizer is supplied with parameters iterator or name,parameter tuple iterator so shared params doesn't get operated more than once
- If parameter is stored in ParameterList then the parameter name will be variable_name.index
- name of the parameter depends on at which level .named_parameters gets called
- Pytorch ignores underscore in variable names and it doesn't mean they will not be in parameters collection.

## Arch params

Imagine we have N arch parameters of K kinds, each kind having count N_k. Each parameter is one tensor. Some of the parameters may reside in
ops, some may be at cell or model level. Their names are determined by where they reside. For example, cell level arch param might get named at model level as cells.0.arch_param1. Some of these parameters may get shared in different parts of the model.

So we need ways to retrieve parameters by:
- their kind
- by owner
- Are they shared or owned

This can be achieved by naming convention for variables where such parameters will be stored. Let's define this convention as kind_arch_param. This way any parameter with name ending in _arch_param is considered as architecture parameter. Their full name in the form module1.module2.kind1_arch_param defines where they reside. The part after last "." and without _arch_param suffix defines the kind of the parameter. While Pytorch automatically avoids double listing for shared parameters, a module can have following convention to keep things clean: Module keeps arch parameters in dictionary where key is same as what their variable names would have been. This way Pytorch doesn't register them automatically. If module does own these parameters, it will create variables with same name so they get registered. Module then can provide following methods: get_owned_params, get_shared_params, is_owned_param(p). For parameter sharing, module may receive dictionary of parameters owned by someone else and given module can decide to share some or all of those.

So, we stipulate that each instance of nn.Module type have same number and type of arch params. So Cell may have one set of arch params, Op have another, Model have another and so on. Question is (1) is it possible to share only subset of one's parameters among instances? (2) how Cell1 can share its arch parameters with Cell2 and Cell3 and Cell4 can with Cell5, Cell6. I think supporting this level of infinite flexibility can potentially make things complex. So let's see how we can do subset of these functionalities. We will have ModelDescBuilder decide which module shares arch params with which one. This can be done with base *Desc object having  member specifying identity of object it will receive parameters from. If no arch parameter is received then object shall create its own. If it did, it may take whole or portion of it and create rest of its own. One can access arch_params method to access params for that module directly and pass parameter recursive=True to get arch params of entire module hierarchy. The return value is ArchParams object.