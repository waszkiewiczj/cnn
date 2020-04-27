import torchvision
import torchvision.transforms as transforms


def train_transform(input_size):
    return transforms.Compose(
        [transforms.RandomHorizontalFlip(),
        #  transforms.RandomCrop(32, padding=4),
        #  transforms.RandomRotation((-10, 10)),
         transforms.Resize(input_size),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


def test_transform(input_size):
    return transforms.Compose(
        [transforms.Resize(input_size),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


def from_torch(train=True, input_size=32):
    transform = test_transform(input_size)
    return torchvision.datasets.CIFAR10(root='./cifar-dataset/torch', train=train, download=True, transform=transform)


def from_kaggle(train=True, input_size=32):
    if train:
        transform = train_transform(input_size)
        return torchvision.datasets.ImageFolder(root='./cifar-dataset/kaggle/train', transform=transform)
    else:
        transform = test_transform(input_size)
        return torchvision.datasets.ImageFolder(root='./cifar-dataset/kaggle/test', transform=transform)
