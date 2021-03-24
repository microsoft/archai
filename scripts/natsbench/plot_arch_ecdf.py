# this is installed via pip
from nats_bench import create

from archai.common.utils import search_space_dataset_error_cdf

import os
import plotly.graph_objects as go

def main():

    # Create the API instance for topology search space in natsbench
    api = create('C:\\Users\\dedey\\dataroot\\natsbench\\NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=True)
    savedir = 'D:\\archai_experiment_reports'

    datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
    ecdfs = {}
    for dataset in datasets:
        test_errs = []
        for arch_id in range(len(api)):
            info = api.get_more_info(arch_id, dataset, hp=200, is_random=False)
            test_accuracy = info['test-accuracy']
            test_errs.append((100.0 - test_accuracy)/100.0)
        err_ratio_tuples = search_space_dataset_error_cdf(test_errs)
        ecdfs[dataset] = err_ratio_tuples

    fig = go.Figure()
    for name, ecdf in ecdfs.items():
        x = [e[0] for e in ecdf]
        y = [e[1] for e in ecdf]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name))
    fig.update_layout(title='Error CDFs Natsbench Topology Space', 
                    xaxis_title='Test Error', 
                    yaxis_title='Share of architectures with error below value')
    savename = os.path.join(savedir, 'natsbench_tss_ecdfs.html')
    fig.write_html(savename)
    fig.show()


if __name__ == '__main__':
    main()