from torch import rand
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from LFnet import LFnet


def get_dataloaders():
    train = MNIST(root="./data/train", train=True, download=True)
    test = MNIST(root="./data/test", train=False, download=True)

    train_dataloader = DataLoader(train, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    train_dl, test_dl = get_dataloaders()
    model = LFnet()
    model(rand(1, 3, 36, 36))
    print(model)
    model.add_layer(type="Conv2D", kernel_size=(3, 3), in_channels=3, out_channels=6, stride=(2,2))
    model(rand(1, 3, 36, 36))
    print(model)
    model.add_layer(type="Conv2D", kernel_size=(3, 3), in_channels=6, out_channels=12)
    model(rand(1, 3, 36, 36))
    print(model)
