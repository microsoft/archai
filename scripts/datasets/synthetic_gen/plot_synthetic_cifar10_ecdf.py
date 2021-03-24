from archai.common.utils import search_space_dataset_error_cdf

import os
import plotly.graph_objects as go
import yaml

def main():

    arch_testerror_file = 'D:\\archai_experiment_reports\\nb_reg_b256_e200_sc10\\arch_id_test_accuracy_synthetic_cifar10.yaml'
    savedir = 'D:\\archai_experiment_reports'

    # load data
    with open(arch_testerror_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.Loader)

    test_errs = [v for _, v in data.items()]
    ecdf = search_space_dataset_error_cdf(test_errs)

    fig = go.Figure()
    x = [e[0] for e in ecdf]
    y = [e[1] for e in ecdf]
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='synthetic_cifar10'))
    fig.update_layout(title='Error CDFs Natsbench Topology Space on Synthetic Cifar10', 
                    xaxis_title='Test Error', 
                    yaxis_title='Share of architectures with error below value')
    savename = os.path.join(savedir, 'natsbench_tss_synthetic_cifar10_ecdf.html')
    fig.write_html(savename)
    fig.show()


if __name__ == '__main__':
    main()