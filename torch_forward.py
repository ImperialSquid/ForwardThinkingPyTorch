from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn, rand


def get_dataloaders():
    train = MNIST(root="./data/train", train=True, download=True)
    test = MNIST(root="./data/test", train=False, download=True)

    train_dataloader = DataLoader(train, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader


def make_model():
    class LFnet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = []
            self.layers.append(nn.Identity())  # use Identity layer as place holder until layers get added
            self.model = nn.Sequential(*self.layers)

        def forward(self, x):
            return self.model(x)

        def add_layer(self, **kwargs):
            # freeze all pre existing layers
            for layer in self.layers:
                layer.requires_grad_(requires_grad=False)

            # remove type from kwargs to prevent unexpected argument error
            type_ = kwargs["type"]
            layer_kwargs = {key: kwargs[key] for key in kwargs if key != "type"}

            if type_ == "Conv2D":
                self.layers.append(nn.Conv2d(**layer_kwargs))
            elif type_ == "Linear":
                self.layers.append(nn.Linear(**layer_kwargs))

            self.model = nn.Sequential(* self.layers)

    return LFnet()


if __name__ == "__main__":
    train_dl, test_dl = get_dataloaders()
    model = make_model()
    model(rand(1, 3, 36, 36))
    print(model)
    model.add_layer(type="Conv2D", kernel_size=(3, 3), in_channels=3, out_channels=6)
    model(rand(1, 3, 36, 36))
    print(model)
    model.add_layer(type="Conv2D", kernel_size=(3, 3), in_channels=6, out_channels=12)
    model(rand(1, 3, 36, 36))
    print(model)
