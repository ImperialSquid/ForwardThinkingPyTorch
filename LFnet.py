from torch import nn


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