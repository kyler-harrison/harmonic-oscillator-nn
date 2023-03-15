import torch 
import torch.nn as nn
import torch.optim as optim


class FeedFwdNN(nn.Module):

    def __init__(self, input_size, output_size, optim_lr=0.0001):
        super(FeedFwdNN, self).__init__()

        # network, starting with simple architecture
        self.dense1 = nn.Linear(input_size, 16)
        self.relu1 = nn.ReLU()
        self.out_dense = nn.Linear(16, output_size)

        # optimizer and loss
        self.optimizer = optim.Adam(self.parameters(), lr=optim_lr)
        self.criteria = nn.MSELoss()
        self.loss = None

    def forward(self, input_data):
        """
        Description:
        Forward pass of the neural network.

        Inputs:
        input_data: tensor of a single sample of data

        Ouput:
        returns tensor of network's final output
        """
        x = self.dense1(input_data)
        x = self.relu1(x)
        x = self.out_dense(x)
        return x

    def backward(self, outputs, targets):
        """
        Description:
        Backward pass of the neural network. Loss computed, optimizer
        zeroed, loss and optimizer complete backward step.

        Inputs:
        outputs: tensor of network's predicted outputs of a given sample
                 (i.e. output of forward pass)
        targets: actual output tensor for given sample

        Output:
        returns loss value between outputs and targets
        """
        self.loss = self.criteria(outputs, targets)
        loss_copy = self.loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return loss_copy

    def compute_loss(self, outputs, targets):
        """
        Description:
        Computes loss without actually backpropagating. Used for checking
        network's progress with test data.

        Inputs:
        outputs: tensor of network's predicted outputs of a given sample
                 (i.e. output of forward pass)
        targets: actual output tensor for given sample

        Output:
        returns loss value between outputs and targets
        """
        return self.criteria(outputs, targets)


