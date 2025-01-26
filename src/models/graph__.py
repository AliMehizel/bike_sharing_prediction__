import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GCNModel(nn.Module):
    """
    GCNModel implements a Graph Convolutional Network (GCN) for graph-based data
    using the PyTorch Geometric library.
    
    Attributes:
        conv1 (GCNConv): First graph convolutional layer.
        conv2 (GCNConv): Second graph convolutional layer.
        fc (nn.Linear): Fully connected layer to produce the final output.
    
    Methods:
        forward(x, edge_index): Defines the forward pass of the model.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes the GCNModel class.
        
        Args:
            input_dim (int): The number of features for each node in the graph.
            hidden_dim (int): The number of hidden units in the hidden layer.
            output_dim (int): The number of output units (final output dimension).
        """
        super(GCNModel, self).__init__()
        
   
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        

        self.fc = nn.Linear(output_dim, 1)
        
    def forward(self, x, edge_index):
        """
        Defines the forward pass of the GCN model.
        
        Args:
            x (Tensor): Node features of shape (num_nodes, input_dim).
            edge_index (Tensor): The edge indices representing the graph's connectivity.
        
        Returns:
            Tensor: Final output after passing through the layers.
        """

        x = torch.relu(self.conv1(x, edge_index))
        
        x = torch.relu(self.conv2(x, edge_index))
    
        x = torch.mean(x, dim=0)
        

        x = self.fc(x)
        
        return x

