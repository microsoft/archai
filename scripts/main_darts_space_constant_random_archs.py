# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import subprocess
import os

from archai.common.utils import exec_shell_command

def main():
    parser = argparse.ArgumentParser(description='Darts Space Constant Random Wrapper Main')
    parser.add_argument('--algos', type=str, default='darts_space_constant_random_archs',
                        help='NAS algos to run, separated by comma')
    parser.add_argument('--arch-list-index', type=int)
    parser.add_argument('--num-archs', type=int)
    parser.add_argument('--datasets', type=str, default='cifar10')
    args, extra_args = parser.parse_known_args()

    # hard coded list of architectures to process
    all_archs = list(range(0,1000))
    archs_to_proc = all_archs[args.arch_list_index:args.arch_list_index+args.num_archs]

    for arch_id in archs_to_proc:
        # assemble command string    
        print(os.getcwd())
        print(os.listdir('.'))
        
        command_list = ['python', 'scripts/main.py', '--full', '--algos', f'{args.algos}',\
                        '--common.seed', '36', '--nas.eval.dartsspace.arch_index', f'{arch_id}',\
                        '--exp-prefix', f'darts_space_constant_random_archs_{arch_id}', '--datasets', f'{args.datasets}']
        
        print(command_list)
        ret = subprocess.run(command_list)



if __name__ == '__main__':
    main()