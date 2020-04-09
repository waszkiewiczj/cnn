import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def from_torch(train=True):
    global transform
    return torchvision.datasets.CIFAR10(root='./cifar-dataset/torch', train=train, download=True, transform=transform)


def from_kaggle(train=True):
    global transform
    if train:
        return torchvision.datasets.ImageFolder(root='./cifar-dataset/kaggle/train', transform=transform)
    else:
        return torchvision.datasets.ImageFolder(root='./cifar-dataset/kaggle/test', transform=transform)