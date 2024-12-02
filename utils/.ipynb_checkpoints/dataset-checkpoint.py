import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

def get_cifar10_loader(batch_size):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data/cifar10', download=True, transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader