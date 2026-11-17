import torch.nn as nn

COEFFICIENT_LAYERS = (10, 400, 400, 400, 400, 3)

class NN_MLP(nn.Module):
    def __init__(self, layer_info):
        super(NN_MLP, self).__init__()
        self.net = nn.Sequential()
        n = len(layer_info) - 1
        for i in range(n - 1):
            self.net.add_module('linear_layer_%d' %(i), nn.Linear(layer_info[i], layer_info[i + 1]))
            self.net.add_module('gelu_layer_%d' %(i), nn.GELU())
        self.net.add_module('linear_layer_%d' %(n - 1), nn.Linear(layer_info[n - 1], layer_info[n]))

    def forward(self, x):
        return self.net(x)


