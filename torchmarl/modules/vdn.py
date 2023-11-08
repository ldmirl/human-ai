"""
Value Decomposition Networks
Paper: https://arxiv.org/pdf/1706.05296.pdf
"""
import copy
from functools import partial
import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchmarl.networks import GRU
from torchmarl.buffers import ReplayBuffer
from torchmarl.utils import epsilon_greedy

# Supports VDN & VDN w/ state
class VDNNet(nn.Module):
  def __init__(
      self, 
      use_state: bool = False, 
      in_dim: int = None,
      mixer_embed_dim: int = None,
      n_agents: int = None,
      *args, **kwargs
  ) -> None:
    super(VDNNet, self).__init__()
    self.use_state = use_state
    if self.use_state:
      self.v = nn.Sequential(
        nn.Linear(in_dim, mixer_embed_dim),
        nn.ReLU(),
        nn.Linear(mixer_embed_dim, 1)
      )
      self.in_dim = in_dim
      self.n_agents = n_agents

  def forward(self, q: torch.Tensor, state: torch.Tensor = None):
    if not self.use_state:
      # Perform normal value decomposition as specified https://arxiv.org/pdf/1706.05296.pdf
      return torch.sum(q, dim=-1, keepdim=True)

    # https://arxiv.org/abs/1803.11485
    batch_size = q.shape[0] # q: (batch_size, n_agents, n_actions)
    q = q.reshape(-1, 1, self.n_agents)
    state = state.reshape(-1, self.in_dim) # state: (batch_size, in_dim)
    v = self.v(state).view(-1, 1, 1)
    out = torch.sum(q, dim=-1, keepdim=True) + v
    return out.view(batch_size, -1, 1)


class VDN:
  def __init__(
    self, 
    in_dim, 
    h_dim, 
    out_dim,
    n_agents,
    lr: float = 5e-4, 
    gamma: float = 0.99,
    grad_norm_clip: int = 10, 
    buffer_maxsize: int = 5000, 
    batch_size: int = 32, 
    optim_alpha: float = 0.99, 
    optim_eps: float = 1e-5, 
    eps_max: float = 1., 
    eps_min: float = 0.5, 
    eps_decay: int = 50_000, 
    double_q: bool = True, 
    target_update_freq: int = 200,
    use_state: bool = False,
    state_shape: int = None,
    embed_dim: int = 64,
    device: str = "cpu", 
  ) -> None:
    self.n_agents = n_agents
    self.network = GRU(in_dim, h_dim, out_dim).to(device)
    self.target_network = copy.deepcopy(self.network)
    self.mixer = VDNNet(use_state, state_shape, embed_dim, n_agents).to(device)
    self.target_mixer = copy.deepcopy(self.mixer)
    self.parameters = list(self.network.parameters()) + list(self.mixer.parameters())
    self.optim = optim.RMSprop(params=self.parameters, lr=lr, alpha=optim_alpha, eps=optim_eps)
    self.buffer = ReplayBuffer(size=buffer_maxsize, device=device)

    self._epsilon_greedy = partial(epsilon_greedy, eps_max=eps_max, eps_min=eps_min, eps_decay=eps_decay)
    self._eps = eps_max

    self.batch_size = batch_size
    self.gamma = gamma
    self.grad_norm_clip = grad_norm_clip
    self.double_q = double_q
    self.device = device
    self.target_update_freq = target_update_freq
    
    self._prev_log_step = 0
    self._prev_update_target_step = 0
  
  def init_hidden(self, batch_size=1):
    self._hx_state = self.network.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
    return self._hx_state
  
  def select_actions(self, timestep, num_steps, test):
    observation = torch.tensor(timestep.observation, dtype=torch.float32).to(self.device)
    out, self._hx_state = self.network(observation, self._hx_state)
    out = out.view(timestep.observation.shape[0], self.n_agents, -1)
    actions, self._eps = self._epsilon_greedy(out, torch.tensor(timestep.avail_actions), num_steps=num_steps, test=test)
    return actions
  
  def update(self, num_episodes, num_steps):
    if self.buffer.can_sample(self.batch_size): 
      batch = self.buffer.sample(self.batch_size)
      observation, actions, avail_actions, rewards, terminated, mask = (
        batch["observations"], batch["actions"][:, :-1], 
        batch["avail_actions"], batch["rewards"][:, :-1], 
        batch["terminated"][:, :-1], batch["__mask__"][:, :-1]
      )
      mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
      batch_size, sequence_length = batch.batch_size, batch.sequence_length
      n_agents = self.n_agents
      
      # Calculate estimated q-values
      network_out = []
      hidden_state = self.init_hidden(batch_size)
      for t in range(sequence_length):
        q, hidden_state = self.network(observation[:, t], hidden_state)
        q = q.reshape(batch_size, n_agents, -1)
        network_out.append(q)
      network_out = torch.stack(network_out, dim=1) # Concat over time
      # Pick the q-values for the actions taken by the agent
      chosen_actions_qvalues = torch.gather(network_out[:, :-1], dim=3, index=actions).squeeze(3) # Remove the last dim
      
      # Calculate the q-values necessary for target
      target_network_out = []
      target_hidden_state = self.init_hidden(batch_size)
      for t in range(sequence_length):
        target_q, target_hidden_state = self.target_network(observation[:, t], target_hidden_state)
        target_q = target_q.reshape(batch_size, n_agents, -1)
        target_network_out.append(target_q)
      # We don't need the first timesteps q-values estimate for calculating targets
      target_network_out = torch.stack(target_network_out[1:], dim=1) # Concat across time
      # Mask out unavailable actions
      target_network_out[avail_actions[:, 1:] == 0.] = -9999999
      
      # Max over target q-values
      if self.double_q:
        # Get actions that maximise live Q (for double q-learning)
        net_out_detach = network_out.clone().detach()
        net_out_detach[avail_actions == 0.] = -9999999
        max_actions = net_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
        target_max_qvalues = torch.gather(target_network_out, 3, max_actions).squeeze(3)
      else:
        target_max_qvalues = target_network_out.max(dim=3)[0]
      
      chosen_actions_qvalues = self.mixer(chosen_actions_qvalues, batch['state'][:, :-1])
      target_max_qvalues = self.target_mixer(target_max_qvalues, batch['state'][:, 1:])

      # print(rewards.shape, terminated.shape, target_max_qvalues.shape, chosen_actions_qvalues.shape)
      # Calculate 1-step q-learning targets
      targets = rewards + self.gamma * (1 - terminated) * target_max_qvalues
      # td-error
      td_error = (chosen_actions_qvalues - targets.detach())
      # Expand the mask to match td-error size
      mask = mask.expand_as(td_error)
      # 0-out the targets that came from padded data
      masked_td_error = td_error * mask
      # Normal L2 loss, take mean over actual data
      loss = (masked_td_error ** 2).sum() / mask.sum()

      # Optimize
      self.optim.zero_grad()
      loss.backward()
      grad_norm = nn.utils.clip_grad_norm_(self.parameters, self.grad_norm_clip)
      self.optim.step()

      if (num_episodes - self._prev_update_target_step) / self.target_update_freq >= 1.0:
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self._prev_update_target_step = num_episodes

      mask_elem = mask.sum().item() 
      results = {
        "loss": loss.cpu().item(),
        "grad_norm": grad_norm.cpu().item(),
        "td_error_abs": (masked_td_error.abs().sum().item()/mask_elem),
        "q_taken_mean": (chosen_actions_qvalues * mask).sum().item()/(mask_elem * n_agents),
        "target_mean": (targets * mask).sum().item()/(mask_elem * n_agents),
        "epsilon": self._eps
      }
      return results
    return {"epsilon": self._eps}
  
  def load(self, path, episode, step):
    f = os.path.join(path, f"{episode}_{step}_vdn_params.pkl")
    self.network.load_state_dict(torch.load(f))
  
  def save(self, path, episode, step):
    torch.save(
      self.network.state_dict(), 
      os.path.join(path, f"{episode}_{step}_vdn_params.pkl")
    )