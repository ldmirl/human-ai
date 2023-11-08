import argparse
import numpy as np
import pathlib
import random
import torch

def parse_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
  parser.add_argument("--root-dir", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1] / "results")
  parser.add_argument("--num-steps", type=int, default=2_050_000)
  parser.add_argument("--num-episodes", type=int, default=0)
  parser.add_argument("--test-episodes", type=int, default=32)
  parser.add_argument("--test-interval", type=int, default=10_000)
  parser.add_argument("--log-interval", type=int, default=10_000)
  parser.add_argument("--save-interval", type=int, default=10_000)
  parser.add_argument("--use-wandb", action="store_true", default=False)
  parser.add_argument("--use-tqdm", action="store_true", default=True)
  parser.add_argument("--device-id", type=int, default=0)
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--debug", action="store_true", default=False)
  return parser

def epsilon_greedy(q_values, actions, eps_max, eps_min, eps_decay, num_steps, test=False):
  eps =  max(eps_min, eps_max - ((eps_max - eps_min) / eps_decay) * num_steps) if not test else 0.0
  # mask actions that are excluded from selection
  masked = q_values.clone()
  masked[actions == 0.0] = -float("inf") # should never be selected!
  pick_random = (torch.rand_like(q_values[..., 0]) < eps).long()
  random_actions = torch.distributions.Categorical(actions.float().to(q_values.device)).sample().long()
  actions = pick_random * random_actions + (1 - pick_random) * masked.argmax(dim=-1)
  return actions, eps

def device(device_id: int = None, use_cuda: bool = True, use_mps: bool = True):
  """Returns the device to use for training."""
  if torch.cuda.is_available() and use_cuda:
    return torch.device(f"cuda:{device_id}" if device_id is not None else "cuda")
  if not (use_mps is True and (device_id is not None or device_id != 0)):
    raise ValueError("Device-ID is not available for MPS!")
  try: 
    if torch.backend.mps.is_available() and use_mps: return torch.device("mps")
  except AttributeError: pass
  return torch.device("cpu")

def seed(seed:int = None) -> int:
  if seed is None: seed = random.randint(1, int(1e9)) 
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  return seed