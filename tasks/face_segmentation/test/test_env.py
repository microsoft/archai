import archai
import os

print(archai.__file__)

folder = os.path.dirname(archai.__file__)
common = os.path.join(folder, 'common')

with open(os.path.join(common, 'store.py'), 'r') as f:
    print(''.join(f.readlines()))
