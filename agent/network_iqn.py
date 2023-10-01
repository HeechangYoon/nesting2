import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NoisyLinear(nn.Linear):
    # Noisy Linear Layer for independent Gaussian Noise
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        # make the sigmas trainable:
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        # not trainable tensor for the nn.Module
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        # extra parameter for the bias and register buffer for the bias parameter
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))

        # reset parameter as initialization of the layer
        self.reset_parameter()

    def reset_parameter(self):
        """
        initialize the parameter of the layer and bias
        """
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input, noisy=True):
        # sample random noise in sigma weight buffer and bias buffer
        if noisy:
            self.epsilon_weight.normal_()
            weight = self.weight + self.sigma_weight * self.epsilon_weight
            bias = self.bias
            if bias is not None:
                self.epsilon_bias.normal_()
                bias = bias + self.sigma_bias * self.epsilon_bias
        else:
            weight = self.weight
            bias = self.bias
        return F.linear(input, weight, bias)


class Network(nn.Module):
    def __init__(self, state_size, x_action_size, a_action_size, N):
        super(Network, self).__init__()
        self.state_size = state_size
        self.x_action_size = x_action_size
        self.a_action_size = a_action_size

        self.N = N
        self.n_cos = 64
        self.pis = torch.FloatTensor([np.pi * i for i in range(1, self.n_cos + 1)]).view(1, 1, self.n_cos).to(device)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cos_embedding = nn.Linear(self.n_cos, 52 * 67 * 32)

        self.fc1_x = NoisyLinear(52 * 67 * 32, 1024)
        self.fc2_x = NoisyLinear(1024, 512)
        self.fc3_x = NoisyLinear(512, 256)

        self.fc1_a = NoisyLinear(52 * 67 * 32, 1024)
        self.fc2_a = NoisyLinear(1024, 512)
        self.fc3_a = NoisyLinear(512, 256)

        self.advantage_x = NoisyLinear(256, x_action_size)
        self.value_x = NoisyLinear(256, 1)
        self.advantage_a = NoisyLinear(256, a_action_size)
        self.value_a = NoisyLinear(256, 1)

    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1).to(device)  # (batch_size, n_tau, 1)  .to(self.device)
        cos = torch.cos(taus * self.pis)

        assert cos.shape == (batch_size, n_tau, self.n_cos), "cos shape is incorrect"
        return cos, taus

    def forward_x(self, x, num_tau=8, noisy=True):
        batch_size = x.size(0)

        h = F.leaky_relu(self.conv1(x))
        h = self.pool1(h)
        h = F.leaky_relu(self.conv2(h))
        h = self.pool2(h)
        h = h.contiguous().view(-1, 52 * 67 * 32)

        cos, taus = self.calc_cos(batch_size, num_tau)  # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size * num_tau, self.n_cos)
        cos_h = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, 52 * 67 * 32)  # (batch, n_tau, layer)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        h = (h.unsqueeze(1) * cos_h).view(batch_size * num_tau, 52 * 67 * 32)

        h = torch.selu(self.fc1_x(h, noisy=noisy))
        h = torch.selu(self.fc2_x(h, noisy=noisy))
        h = torch.selu(self.fc3_x(h, noisy=noisy))
        advantage = self.advantage_x(h, noisy=noisy)
        value = self.value_x(h, noisy=noisy)

        out = value + advantage - advantage.mean(dim=1, keepdim=True)
        out = out.view(batch_size, num_tau, self.x_action_size)

        return out, taus

    def forward_a(self, x, num_tau=8, noisy=True):
        batch_size = x.size(0)

        h = F.leaky_relu(self.conv1(x))
        h = self.pool1(h)
        h = F.leaky_relu(self.conv2(h))
        h = self.pool2(h)
        h = h.contiguous().view(-1, 52 * 67 * 32)

        cos, taus = self.calc_cos(batch_size, num_tau)  # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size * num_tau, self.n_cos)
        cos_h = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, 52 * 67 * 32)  # (batch, n_tau, layer)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        h = (h.unsqueeze(1) * cos_h).view(batch_size * num_tau, 52 * 67 * 32)

        h = torch.selu(self.fc1_a(h, noisy=noisy))
        h = torch.selu(self.fc2_a(h, noisy=noisy))
        h = torch.selu(self.fc3_a(h, noisy=noisy))
        advantage = self.advantage_a(h, noisy=noisy)
        value = self.value_a(h, noisy=noisy)

        out = value + advantage - advantage.mean(dim=1, keepdim=True)
        out = out.view(batch_size, num_tau, self.a_action_size)

        return out, taus

    def get_qvalues_x(self, inputs, noisy=True):
        quantiles, _ = self.forward_x(inputs, self.N, noisy=noisy)
        actions = quantiles.mean(dim=1)
        return actions

    def get_qvalues_a(self, inputs, noisy=True):
        quantiles, _ = self.forward_a(inputs, self.N, noisy=noisy)
        actions = quantiles.mean(dim=1)
        return actions