import yaml
from archai.common.yaml_utils import resolve_all

s="""
common:
    dataset:
        name: 'd1'
        classes: 3
        cifar:
            limit: -1
    logging: True

autoaug:
    dataset: '_copy: /nas/train/dataset'
    logging: '_copy: /nas/logging'

nas:
    train:
        dataset:
            _copy: '/common/dataset'
            classes: 4
            cifar:
                workers: 0
    logging: '_copy: /common/logging'
"""

d = yaml.safe_load(s)
print(d)
resolve_all(d)
print(yaml.dump(d))