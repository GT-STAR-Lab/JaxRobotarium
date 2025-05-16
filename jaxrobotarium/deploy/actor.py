import torch
import torch.nn as nn

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Initialize weights
        self.W_z = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.U_z = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        
        self.W_r = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.U_r = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        
        self.W_h = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.U_h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, x, h_prev):
        r_t = torch.sigmoid(x @ self.W_r + h_prev @ self.U_r + self.b_r)
        z_t = torch.sigmoid(x @ self.W_z + h_prev @ self.U_z + self.b_z)
        
        h_tilde = torch.tanh((x @ self.W_h + self.b_ih) + r_t * (h_prev @ self.U_h + self.b_hh))
        h_t = (1 - z_t) * h_tilde + z_t * h_prev
        
        return h_t

class RNNActorQMix(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.Dense_0 = nn.Linear(input_dim, hidden_dim)
        self.GRUCell_0 = GRUCell(hidden_dim, hidden_dim)
        self.Dense_1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden): 
        embedding = self.Dense_0(input)
        embedding = torch.relu(embedding)
        hidden = self.GRUCell_0(embedding, hidden)
        output = self.Dense_1(hidden)

        return output, hidden
    
class RNNActorPQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.normalize = nn.LayerNorm(hidden_dim)

        self.Dense_0 = nn.Linear(input_dim, hidden_dim)
        self.Dense_1 = nn.Linear(hidden_dim, hidden_dim)
        self.GRUCell_0 = GRUCell(hidden_dim, hidden_dim)
        self.Dense_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden): 
        embedding = self.Dense_0(input)
        embedding = self.normalize(embedding)
        embedding = torch.relu(embedding)
        
        embedding = self.Dense_1(embedding)
        embedding = self.normalize(embedding)
        embedding = torch.relu(embedding)

        hidden = self.GRUCell_0(embedding, hidden)
        output = self.Dense_2(hidden)

        return output, hidden

class RNNActorMAPPO(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.Dense_0 = nn.Linear(input_dim, hidden_dim)
        self.GRUCell_1 = GRUCell(hidden_dim, hidden_dim)
        self.Dense_1 = nn.Linear(hidden_dim, hidden_dim)
        self.Dense_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden): 
        embedding = self.Dense_0(input)
        embedding = torch.relu(embedding)
        hidden = self.GRUCell_1(embedding, hidden)
        embedding = self.Dense_1(hidden)
        embedding = torch.relu(embedding)
        output = self.Dense_2(embedding)

        return output, hidden

class RNNActorIPPO(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.Dense_0 = nn.Linear(input_dim, hidden_dim)
        self.GRUCell_1 = GRUCell(hidden_dim, hidden_dim)
        self.Dense_1 = nn.Linear(hidden_dim, hidden_dim)
        self.Dense_2 = nn.Linear(hidden_dim, output_dim)

        # unused critic params
        self.Dense_3 = nn.Linear(hidden_dim, hidden_dim)
        self.Dense_4 = nn.Linear(hidden_dim, 1)

    def forward(self, input, hidden): 
        embedding = self.Dense_0(input)
        embedding = torch.relu(embedding)
        hidden = self.GRUCell_1(embedding, hidden)
        embedding = self.Dense_1(hidden)
        embedding = torch.relu(embedding)
        output = self.Dense_2(embedding)

        return output, hidden