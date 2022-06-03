# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# uniquify and sort the names list.
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
fruits = []
name_file = os.path.join(script_dir, 'names.txt')
with open(name_file, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        if line and line not in fruits:
            fruits += [line]

fruits.sort()
with open(name_file, 'w') as f:
    f.write('\n'.join(fruits))

print(f"saved {len(fruits)} names")
