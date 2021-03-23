# this is installed via pip
from nats_bench import create

from archai.common.utils import search_space_dataset_error_cdf

import plotly.express as px

def main():

    # Create the API instance for topology search space in natsbench
    api = create('C:\\Users\\dedey\\dataroot\\natsbench\\NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=True)

    dataset_name = 'cifar10'
    test_errs = []
    for arch_id in range(0, 100):
        info = api.get_more_info(arch_id, dataset_name, hp=200, is_random=False)
        test_accuracy = info['test-accuracy']
        test_errs.append((100.0 - test_accuracy)/100.0)
    
    err_ratio_tuples = search_space_dataset_error_cdf(test_errs)
    x = [err_ratio[0] for err_ratio in err_ratio_tuples]
    y = [err_ratio[1] for err_ratio in err_ratio_tuples]
    fig = px.line(x=x, y=y)
    fig.show()
    print('dummy')


if __name__ == '__main__':
    main()