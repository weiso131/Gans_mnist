import torch.nn as nn

class Mnist_classifer(nn.Module):
    def __init__(self):
        super(Mnist_classifer, self).__init__()
        self.output = nn.Sequential(nn.Linear(784, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(),
                                    nn.Linear(1024, 256),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 10)
                                    )
    def forward(self, x):
        x = x.view(-1, 784)
        return self.output(x)