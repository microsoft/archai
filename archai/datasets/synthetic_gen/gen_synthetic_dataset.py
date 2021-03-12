import torch
import torch.nn as nn

import math as ma

from archai.datasets.synthetic_gen.simple_nn import SimpleNN

def main():
    
    # conf
    img_shape = [32, 32, 3]
    n_classes = 10
    n_examples = 60000
    n_train = 50000
    seed = 42
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

    for i in range(n_examples):
        
        # sample a gaussian image
        sampled_img = torch.normal(mean=mean, std=std).reshape(img_shape)

        # pass image through a randomly initialized neural network
        logits = net(sampled_img.view(-1))

        # compute the groundtruth class for it
        gt_class = torch.argmax(logits)

        print(f'Gt class {gt_class.item()}')

        







if __name__ == '__main__':
    main()