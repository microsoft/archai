Food-101
========

The Food-101 dataset consists of 101 food categories, each containing a total of 101,000 images. Each class has 250 manually reviewed test images and 750 training images. The training images have not been cleaned and may contain noise in the form of intense colors and incorrect labels. All images have been rescaled to have a maximum side length of 512 pixels.

Transforms
----------

Variation 1
^^^^^^^^^^^

.. code-block:: python

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

Variation 2
^^^^^^^^^^^

.. code-block:: python

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(self.imgenet_mean, self.imgenet_std)
    ])

    valid_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(self.imgenet_mean, self.imgenet_std)
    ])

Source: https://nbviewer.jupyter.org/github/shubhajitml/food-101/blob/master/food-101-pytorch.ipynb

Variation 3
^^^^^^^^^^^

.. code-block:: python

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

Source: https://github.com/dashimaki360/food101/blob/master/src/train.py

State-of-The-Art
----------------

The following table presents the state-of-the-art results for the Food-101 dataset, a collection of images of food items. These results represent the current best performance of various machine learning algorithms on this dataset, and provide a benchmark for comparing the performance of new algorithms. The table includes the name of the method, its top-1 and top-5 accuracy, and the publication in which the method was introduced.

.. list-table::
    :header-rows: 1

    *   - Method
        - Top - 1
        - Top - 5
        - Publication
    *   - HoG
        - 8.85
        - \-
        - ECCV2014
    *   - SURF BoW-1024
        - 33.47
        - \-
        - ECCV2014
    *   - SURF IFV-64
        - 44.79
        - \-
        - ECCV2014
    *   - SURF IFV-64 + Color Bow-64
        - 49.40
        - \-
        - ECCV2014
    *   - IFV
        - 38.88
        - \-
        - ECCV2014
    *   - RF
        - 37.72
        - \-
        - ECCV2014
    *   - RCF
        - 28.46
        - \-
        - ECCV2014
    *   - MLDS
        - 42.63
        - \-
        - ECCV2014
    *   - RFDC
        - 50.76
        - \-
        - ECCV2014
    *   - SELC
        - 55.89
        - \-
        - CVIU2016
    *   - AlexNet-CNN
        - 56.40
        - \-
        - ECCV2014
    *   - DCNN-FOOD
        - 70.41
        - \-
        - ICME2015
    *   - DeepFood
        - 77.4
        - 93.7
        - COST2016
    *   - Inception V3
        - 88.28
        - 96.88
        - ECCVW2016
    *   - ResNet-200
        - 88.38
        - 97.85
        - CVPR2016
    *   - WRN
        - 88.72
        - 97.92
        - BMVC2016
    *   - ResNext-101
        - 85.4
        - 96.5
        - Proposed
    *   - WISeR
        - 90.27
        - 98.71
        - UNIUD2016
    *   - **DenseNet - 161**
        - **93.26**
        - **99.01**
        - **Proposed**