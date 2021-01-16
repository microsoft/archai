import pickle

from archai.common import utils

def main():
    in_dataset_file = utils.full_path('~/dataroot/nasbench_ds/nasbench_full.tfrecord.pkl')
    out_dataset_file = utils.full_path('~/dataroot/nasbench_ds/nasbench101_sample.tfrecord.pkl')

    with open(in_dataset_file, 'rb') as f:
      records = pickle.load(f)
    sampled_hashes = set(records[i][0] for i in [0, 4, 40, 400, 4000, 40000, 400000, 401277, len(records)-1])
    sampled = [r for r in records if r[0] in sampled_hashes]
    with open(out_dataset_file, 'wb') as f:
      pickle.dump(sampled, f)

if __name__ == '__main__':
    main()




