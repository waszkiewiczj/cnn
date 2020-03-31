import torchvision
import torchvision.transforms as transforms


def load_CIFAR10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./cifar-dataset', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./cifar-dataset', train=False, download=True, transform=transform)
    return trainset, testset
