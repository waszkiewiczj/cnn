import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def from_torch():
    global transform
    trainset = torchvision.datasets.CIFAR10(root='./cifar-dataset/torch', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./cifar-dataset/torch', train=False, download=True, transform=transform)
    return trainset, testset


def from_kaggle():
    global transform
    trainset = torchvision.datasets.ImageFolder(root='./cifar-dataset/kaggle/train', transform=transform)
    testset = torchvision.datasets.ImageFolder(root='./cifar-dataset/kaggle/test', transform=transform)
    return  trainset, testset