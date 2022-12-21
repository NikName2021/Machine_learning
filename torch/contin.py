from model import Net, network

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision


batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
n_epochs = 10

continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                                momentum=momentum)

# network_state_dict = torch.load('./results/model.pth')
# continued_network.load_state_dict(network_state_dict)
#
# optimizer_state_dict = torch.load('./results/optimizer.pth')
# continued_optimizer.load_state_dict(optimizer_state_dict)