import torchvision
import torchvision.transforms as transforms

def make_transform(input_size):
    return transforms.Compose(
        [
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def from_torch(train=True,input_size=32):
    transform = make_transform(input_size)
    return torchvision.datasets.CIFAR10(root='./cifar-dataset/torch', train=train, download=True, transform=transform)


def from_kaggle(train=True,input_size=32):
    transform = make_transform(input_size)
    if train:
        return torchvision.datasets.ImageFolder(root='./cifar-dataset/kaggle/train', transform=transform)
    else:
        
        return torchvision.datasets.ImageFolder(root='./cifar-dataset/kaggle/test', transform=transform)