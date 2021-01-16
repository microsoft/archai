import logging
import random
from archai.algos.nasbench101.nasbench101_dataset import Nasbench101Dataset
from archai.algos.nasbench101 import model_builder


def main():
  logging.getLogger().setLevel(logging.DEBUG)

  # create dataset
  nsds= Nasbench101Dataset('~/dataroot/nasbench_ds/nasbench_only108.tfrecord.pkl')

  # query for specific model
  data = nsds.query(model_builder.EXAMPLE_DESC_MATRIX, model_builder.EXAMPLE_VERTEX_OPS)
  print('queried', data)

  # sample model
  # nsds is list type object of model statistics
  num_models = len(nsds)
  data = nsds[random.randint(0, num_models-1)]
  print('random', data)

  # nsds is pre-sorted by avg test accuracy
  worst = nsds[0]
  print('worst', worst)

  best = nsds[len(nsds)-1]
  print('best', best)


if __name__ == '__main__':
    main()