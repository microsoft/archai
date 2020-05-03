from archai.common.config import Config

def test_param_override1():
    conf = Config('confs/algos/darts.yaml;confs/datasets/cifar10.yaml')
    assert not conf['nas']['eval']['trainer']['apex']['enabled']
    assert not conf['nas']['eval']['loader']['apex']['enabled']

    conf = Config('confs/algos/darts.yaml;confs/datasets/cifar10.yaml',
           param_args=["--nas.eval.trainer.apex.enabled", "True"])
    assert conf['nas']['eval']['trainer']['apex']['enabled']
    assert conf['nas']['eval']['loader']['apex']['enabled']

def test_param_override2():
    conf = Config('confs/algos/darts.yaml;confs/datasets/imagenet.yaml')
    assert conf['nas']['eval']['trainer']['apex']['distributed_enabled']
    assert conf['nas']['eval']['loader']['apex']['distributed_enabled']

    conf = Config('confs/algos/darts.yaml;confs/datasets/imagenet.yaml',
           param_args=["--nas.eval.trainer.apex.enabled", "True", "--nas.eval.trainer.apex.distributed_enabled", "False"])
    assert not conf['nas']['eval']['trainer']['apex']['distributed_enabled']
    assert not conf['nas']['eval']['loader']['apex']['distributed_enabled']

test_param_override1()
test_param_override2()