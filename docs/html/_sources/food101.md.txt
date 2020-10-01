
## Transofrms

### Variation 1

```
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),ImageNetPolicy(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
```

### Variation 2

From: https://nbviewer.jupyter.org/github/shubhajitml/food-101/blob/master/food-101-pytorch.ipynb
```
        train_tfms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.imgenet_mean, self.imgenet_std)])

        valid_tfms = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.imgenet_mean, self.imgenet_std)])
```

### Variation 3

From: https://github.com/dashimaki360/food101/blob/master/src/train.py
```
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
```

## SOTA

| Method 	| Top - 1  	| Top - 5  	| Publication  	|
|---	|---	|---	|---	|
| HoG    	|8.85   	| - | ECCV2014  	|
|   SURF BoW-1024 	|  33.47  	|   -	| ECCV2014  	|
|   SURF IFV-64 	|  44.79   	|   -	|   ECCV2014 	|
|    SURF IFV-64 + Color Bow-64	|  49.40 	|   -	|   ECCV2014   	|
|   IFV	| 38.88   	| -  	|  ECCV2014  	|
|  RF	|   37.72 	| -  	|   ECCV2014  	|
|   RCF	|   28.46 	| -  	|    ECCV2014	|
|   MLDS 	|    42.63  	| -  	|  ECCV2014	|
|  RFDC	|   50.76   	|  - 	|   ECCV2014 	|
|  SELC 	|     55.89 	|   -	|  CVIU2016 	|
|   AlexNet-CNN 	|  56.40  	|   -	|    ECCV2014	|
|  DCNN-FOOD  	|  70.41  	|   - 	|   ICME2015	|
|   DeepFood 	|   77.4   	|   93.7	|  COST2016 	|
| Inception V3  	|  88.28  	|   96.88 	|   ECCVW2016 	|
|   ResNet-200	|   88.38 	|   	97.85 |    CVPR2016	|
|   WRN 	|   88.72 	|   	 97.92|   BMVC2016	|
|ResNext-101| 85.4|96.5| **Proposed**
|   WISeR 	|   90.27 	|   98.71	|   UNIUD2016 	|
|   **DenseNet - 161**	|  **93.26** 	|   **99.01**	|  **Proposed** 	|

### Model training and SoTA results

From: https://github.com/pyligent/food101-image-classification

- Deep Convolution Neural Network model have achieved remarkable results in image classification problems. For food 101 data the current SoTA results are:
    -  **InceptionV3** : 88.28% / 96.88% (Top 1/Top 5)
    -  **ResNet200** : 90.14% (Top 1)
    -  **WISeR** :  90.27% / 98.71%  (Top 1/Top 5)

- **My Results**: By using the pre-trained ResNet50 model, started by training the network with an image size of 224x224 for 16 epochs , training on image size of 512x512 for additional 16 epochs.
   `top_1_accuracy:`  **89.63%**
   `top_5_accuracy:`  **98.04%**