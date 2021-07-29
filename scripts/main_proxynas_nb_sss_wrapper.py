# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import subprocess
import os
import random

from archai.common.utils import exec_shell_command

def main():
    parser = argparse.ArgumentParser(description='Proxynas SSS Wrapper Main')
    parser.add_argument('--algos', type=str, default='''
                                                        proxynas_natsbench_sss_space,                                                                                                                
                                                    ''',
                        help='NAS algos to run, separated by comma')
    parser.add_argument('--top1-acc-threshold', type=float)
    parser.add_argument('--arch-list-index', type=int)
    parser.add_argument('--num-archs', type=int)
    parser.add_argument('--datasets', type=str, default='cifar10')
    args, extra_args = parser.parse_known_args()

    # hard coded list of architectures to process
    # WARNING: do not edit the numbers here since 
    # they have to be the same each run
    rand_obj = random.Random(36)
    all_archs = rand_obj.sample(range(32768), 1000)
    archs_to_proc = all_archs[args.arch_list_index:args.arch_list_index+args.num_archs]

    for arch_id in archs_to_proc:
        # assemble command string    
        print(os.getcwd())
        print(os.listdir('.'))
        
        command_list = ['python', 'scripts/main.py', '--full', '--algos', f'{args.algos}',\
                        '--common.seed', '36', '--nas.eval.natsbench.arch_index', f'{arch_id}',\
                        '--nas.eval.trainer.top1_acc_threshold', f'{args.top1_acc_threshold}',\
                        '--exp-prefix', f'proxynas_{arch_id}', '--datasets', f'{args.datasets}']
        
        print(command_list)
        ret = subprocess.run(command_list)



if __name__ == '__main__':
    main()