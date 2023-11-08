import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
  def __init__(self, in_dim, h_dim, out_dim):
    super(GRU, self).__init__()
    self.fc1 = nn.Linear(in_dim, h_dim)
    self.rnn = nn.GRUCell(h_dim, h_dim)
    self.fc2 = nn.Linear(h_dim, out_dim)
    self._h_dim = h_dim

  def init_hidden(self): 
    return self.fc1.weight.new(1, self._h_dim).zero_()

  def forward(self, q, hidden_state):
    q = q.reshape(-1, q.shape[-1])
    x = F.relu(self.fc1(q))
    h_in = hidden_state.reshape(-1, self._h_dim)
    h = self.rnn(x, h_in)
    q = self.fc2(h)
    return q, h
