import torch
import torch.nn as nn

import math as ma
from collections import defaultdict
from tqdm import tqdm

from archai.datasets.synthetic_gen.simple_nn import SimpleNN

def main():
    
    # conf
    img_shape = [32, 32, 3]
    n_classes = 10
    n_examples_to_gen = 60000
    max_examples_per_class = 6000
    n_train = 50000
    seed = 42
    out_dir = "C:\\Users\\dedey\\dataroot\\synthetic_cifar10"
    # end conf

    # make torch deterministic
    torch.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # mean and std for gaussian
    num_img_elements = ma.prod(img_shape)
    mean = torch.rand(num_img_elements)
    std = torch.rand(num_img_elements)

    # a simple random neural network to 'label' images
    net = SimpleNN(num_classes = n_classes)

    storage = defaultdict(list)

    for i in tqdm(range(n_examples_to_gen)):
 
        # sample a gaussian image
        sampled_img = torch.normal(mean=mean, std=std).reshape(img_shape)

        # pass image through a randomly initialized neural network
        logits = net(sampled_img.view(-1))

        # compute the groundtruth class for it
        gt_class = torch.argmax(logits)

        # only store examples if max threshold is not exceeded
        if (len(storage[gt_class.item()]) < max_examples_per_class):
            storage[gt_class.item()].append(sampled_img)
        
    
    # print class statistics
    num_examples_all_classes = []
    for key in storage.keys():
        num_examples_this_class = len(storage[key])
        num_examples_all_classes.append(num_examples_this_class)
        print(f'class {key}: {num_examples_this_class}')








if __name__ == '__main__':
    main()