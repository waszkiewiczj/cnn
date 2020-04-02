# Convolutional neural network for CIFAR10

## Datasets

### Loading from PyTorch

Data can be downloaded via **PyTorch** library functions and stored in `.\cifar-dataset\torch\`.

### Loading from Kaggle

Data can be also downloaded from **Kaggle competition** - [CIFAR-10 - Object Recognition in Images](https://www.kaggle.com/c/cifar-10/overview).
This data varies from normal CIFAR-10 - test data consist of extra 290 000 images. No test set image labels are provided.

To use this data in project it must be downloaded from [competition site](https://www.kaggle.com/c/3649/download-all) and unarchived (this refers also to image 7z archives).
To prepare data for further computing, execute script:

    python convert_kaggle_data.py [dir]
    
Where `[dir]` is path to root directory where downloaded data was saved.