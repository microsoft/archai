import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage

import math as ma
from collections import defaultdict
from tqdm import tqdm
import os


def main():
    
    # conf
    img_shape = [32, 32, 3]
    n_classes = 10
    n_examples_to_gen = 500000
    max_examples_per_class = 6000
    n_examples_per_class_train = 5000
    seed = 42
    out_dir = "C:\\Users\\dedey\\dataroot\\synthetic_cifar10"
    # end conf

    # make torch deterministic
    torch.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # linear functions which will generate data labels
    num_img_elements = ma.prod(img_shape)
    ws = torch.randn((n_classes,  num_img_elements))

    # linear function storage
    storage = defaultdict(list)

    for i in tqdm(range(n_examples_to_gen)):
 
        # sample a gaussian image
        sampled_img = torch.randn(img_shape)
        
        # find argmax
        responses = torch.matmul(ws, sampled_img.view(-1, 1))
        gt_class = torch.argmax(responses)

        # only store examples if max threshold is not exceeded
        if (len(storage[gt_class.item()]) < max_examples_per_class):
            storage[gt_class.item()].append(sampled_img)
        
    
    # print class statistics
    num_examples_all_classes = []
    for key in storage.keys():
        num_examples_this_class = len(storage[key])
        num_examples_all_classes.append(num_examples_this_class)
        print(f'class {key}: {num_examples_this_class}')


    # save the dataset to format that pytorch DatasetFolder dataloader
    # can easily plug and play with
    for key in storage.keys():
        train_dir = os.path.join(out_dir, 'train', str(key))
        test_dir = os.path.join(out_dir, 'test', str(key))
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        for i, img in enumerate(storage[key]):
            # to turn it into CHW
            img = img.permute(2, 0, 1)
            if i < n_examples_per_class_train:
                savename = os.path.join(train_dir, str(i)+'.pt')
            else:
                savename = os.path.join(test_dir, str(i)+'.pt')
            torch.save(img, savename)







if __name__ == '__main__':
    main()