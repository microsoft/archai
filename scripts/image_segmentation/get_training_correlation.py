from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

parser = ArgumentParser()
parser.add_argument('--report_path', type=Path, required=True)
parser.add_argument('--save_dir', type=Path, required=True)
parser.add_argument('--metric_name', type=str, default='validation_overall_f1')


if __name__ == '__main__':
    args = parser.parse_args()
    metric = args.metric_name
    assert args.report_path.exists()

    df = pd.read_csv(args.report_path).query('metric == @metric')
    df = df.sort_values(['experiment_name', 'job_name', 'step'])

    # Removes Lovasz fine-tuning
    df = df[~df['best_val_ckpt'].str.contains('/ft_model/', regex=False)]
    df = df.sort_values(['experiment_name', 'job_name', 'step'])

    df['max_f1'] = df.groupby(['experiment_name', 'job_name'])['value'].transform('max')
    df['c_max_f1'] = df.groupby(['experiment_name', 'job_name'])['value'].transform('cummax')

    df['id'] = df['experiment_name'] + df['job_name']

    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    axs_flattened = [ax for ax in axs.flatten()]
    corr = []

    steps = sorted(df['step'].unique().tolist())[0:25]
    
    for i, step in enumerate(steps):
        q = df.query('step == @step')
        corr.append(spearmanr(q['c_max_f1'], q['max_f1']).correlation)
        axs_flattened[i].plot(
            q['c_max_f1'], q['max_f1'], 'o'
        )
        axs_flattened[i].set_title(f'e = {step/5625:.1f}, c = {corr[-1]:.2f}')

    fig.tight_layout()
    fig.savefig(args.save_dir / 'f1_scatter.png')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(np.array(steps[0:30])/5625, corr[0:30], 'o-')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Correlation')
    fig.savefig(args.save_dir / 'f1_correlation.png')

