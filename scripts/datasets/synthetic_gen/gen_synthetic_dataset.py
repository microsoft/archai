import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage

import math as ma
from collections import defaultdict
from tqdm import tqdm
import os

import scripts.datasets.synthetic_gen.simple_nn as simple_nn

def main():
    
    # conf
    img_shape = [32, 32, 3]
    n_classes = 10
    n_examples_to_gen = 400000
    max_examples_per_class = 6000
    n_examples_per_class_train = 5000
    seed = 42
    out_dir = "C:\\Users\\dedey\\dataroot\\synthetic_cifar10"
    # end conf

    # make torch deterministic
    torch.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # a simple random neural network to 'label' images
    net_storage = []
    for i in range(n_classes):
        net_storage.append(simple_nn.SimpleNN(num_classes = n_classes))
    
    storage = defaultdict(list)

    for i in tqdm(range(n_examples_to_gen)):
 
        # sample a gaussian image
        sampled_img = torch.randn(img_shape)

        # pass image through n randomly initialized networks
        response_storage = []
        for j in range(len(net_storage)):
            response = net_storage[j](sampled_img.view(-1))
            response_storage.append(response)

        # compute the groundtruth class for it
        gt_class = torch.argmax(torch.Tensor(response_storage))

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