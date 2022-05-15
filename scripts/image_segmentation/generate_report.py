from typing import Dict, Optional
from argparse import ArgumentParser
from pathlib import Path
import yaml
import traceback
import re

import torch
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from archai.algos.evolution_pareto_image_seg.model import SegmentationNasModel
from archai.algos.evolution_pareto_image_seg.segmentation_trainer import LightningModelWrapper
from archai.algos.evolution_pareto_image_seg.utils import to_onnx

from tqdm import tqdm

parser = ArgumentParser('Runs profiling and parses log files from all job runs in a given directory.')
parser.add_argument('--exp_results_dir', help='Job directory', type=Path, required=True)
parser.add_argument('--reports_dir', help='Job directory', type=Path, required=True)
parser.add_argument('--report_keyword', help='Keyword to filter experiments to save final reports. You can use two keywords using a comma between them.', type=str, default='')
parser.add_argument('--save_step_level', action='store_true', default=False)
parser.add_argument('--overwrite', action='store_true', default=False)
parser.add_argument('--opset_version', type=int, default=11)
parser.add_argument('--ignore_finetuning', action='store_true', default=False)


def tflog2pandas(path: Path) -> pd.DataFrame:
    '''From https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py''' 

    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }

    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})

    try:
        event_acc = EventAccumulator(str(path), DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])

    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()

    return runlog_data


if __name__ == '__main__':
    args = parser.parse_args()

    experiments = [
        d for d in args.exp_results_dir.glob('*')
        if d.is_dir() and (not args.report_keyword or any(kw in str(d) for kw in args.report_keyword.split(',')))
    ]

    print(f'Found {len(experiments)} experiments that match the given keywords:')
    print('\n'.join(['* ' + str(e) for e in experiments]))

    results_df = []
    prefix = 'model' if args.ignore_finetuning else 'ft_model'

    for experiment in tqdm(experiments, desc='Gathering logs from all jobs...'):
        jobs = {
            job: [tflog2pandas(f) for f in (job / f'{prefix}/lightning_logs').glob('*/*tfevents*') if f.exists()]
            for job in experiment.glob('*')
        }
        jobs = {job: df_list for job, df_list in jobs.items() if df_list}

        results_df += [
            pd.concat(job_df).assign(
                experiment_name=lambda x: str(experiment.stem),
                job_name=lambda x: job_path.stem,
                job_path=lambda x: job_path
            ) for job_path, job_df in jobs.items()
        ]
    results_df = pd.concat(results_df)
    results_df['_id'] = results_df['job_name'].str.replace('.*_arc_', '', regex=True)

    # Gets the best validation checkpoint from the `job_path` column
    results_df['best_val_ckpt'] = results_df['job_path'].apply(
        lambda x: list(x.glob(f'{prefix}/best_model/epoch=*.ckpt') if (x / prefix).is_dir() else x.glob('model/best_model/epoch=*.ckpt'))
    )
    results_df = results_df[results_df['best_val_ckpt'].apply(len) > 0]

    # Gets the latest checkpoint
    results_df['best_val_ckpt'] = results_df['best_val_ckpt'].apply(
        lambda x: max(x, key=lambda y: int(re.sub('[^0-9]', '', str(y))))
    )

    # Re-arranges columns for better visualization
    results_df = results_df[['_id', 'experiment_name', 'job_name', 'best_val_ckpt', 'step', 'metric', 'value']]

    # Runs profiling for each model
    op_version = None if args.opset_version == 0 else args.opset_version
    model_results = {}
    for model_ckpt in tqdm(results_df['best_val_ckpt'].unique().tolist(), desc='Profiling models...'):
        print(model_ckpt)
        model = LightningModelWrapper.load_from_checkpoint(str(model_ckpt))
        
        try:
            model_results['model_hash'] = model.model.to_hash()
        except:
            print('Model hash calculation failed. ')
    
        output_path = model_ckpt.parent / (model_ckpt.stem + '.onnx')
        to_onnx(model.model, model_ckpt.parent / (model_ckpt.stem + '.onnx'), img_size=model.hparams['img_size'],
                opset_version=op_version)

    results_df = pd.concat(
        [results_df, results_df['best_val_ckpt'].map(model_results).apply(pd.Series)], axis=1
    )
    results_df['model_hash'] = results_df['model_hash'] if 'model_hash' in results_df.columns else results_df['_id']

    # Prints model-level results
    target_metric = 'validation_overall_f1'

    model_level_results = (results_df
        .query('metric == @target_metric')\
        .groupby(['model_hash', 'experiment_name', 'job_name', 'best_val_ckpt'])\
        ['value'].max()\
        .reset_index()\
        .rename({'value': target_metric}, axis=1)
    )
    args.report_keyword = args.report_keyword + '_' if args.report_keyword else ''
    
    ft_prefix = 'ft' if not args.ignore_finetuning else 'noft'
    model_level_results.to_csv(args.reports_dir / (args.report_keyword + f'{ft_prefix}_model_level.csv'))
    print(args.reports_dir / (args.report_keyword + f'{ft_prefix}_model_level.csv'))

    if args.save_step_level:
        results_df.to_csv(args.reports_dir / (args.report_keyword + f'{ft_prefix}_step_level.csv'))
        print(args.reports_dir / (args.report_keyword + f'{ft_prefix}_step_level.csv'))
