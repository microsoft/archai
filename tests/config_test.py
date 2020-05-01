from archai.common.config import Config

def test_param_override():
    conf = Config('confs/algos/darts.yaml;confs/datasets/cifar10.yaml',
           param_args=["--nas.eval.trainer.apex.enabled", "True"])

    assert conf['nas']['eval']['trainer']['apex']['enabled']

test_param_override()