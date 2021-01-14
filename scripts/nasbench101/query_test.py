from archai.algos.nasbench101.nasbench101_dataset import Nasbench101Dataset
from archai.algos.nasbench101 import model_builder


def main():
  nsds= Nasbench101Dataset('~/dataroot/nasbench_ds/nasbench_only108.tfrecord.pkl')
  data = nsds.query(model_builder.EXAMPLE_DESC_MATRIX, model_builder.EXAMPLE_VERTEX_OPS)
  print(data)

if __name__ == '__main__':
    main()