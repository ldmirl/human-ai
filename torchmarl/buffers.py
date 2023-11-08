from typing import NamedTuple, Any, Dict, Tuple
from dataclasses import dataclass, asdict, make_dataclass, field
import numpy as np
import torch
import torch.nn.functional as F

from dm_env import StepType

class TimeStep(NamedTuple):
  step_type: Any
  reward: Any
  discount: Any
  observation: Any
  avail_actions: Any
  state: Any

  def first(self) -> bool:
    return self.step_type == StepType.FIRST

  def mid(self) -> bool:
    return self.step_type == StepType.MID

  def last(self) -> bool:
    return self.step_type == StepType.LAST
  
# Helper functions for creating TimeStep namedtuples with default settings.

def restart(observations, avail_actions, state):
  """Returns a `TimeStep` with `step_type` set to `StepType.FIRST`."""
  return TimeStep(StepType.FIRST, None, None, observations, avail_actions, state)


def transit(reward, observations, avail_actions, state, discount=1.0):
  """Returns a `TimeStep` with `step_type` set to `StepType.MID`."""
  return TimeStep(StepType.MID, reward, discount, observations, avail_actions, state)


def terminate(reward, observations, avail_actions, state, discount=1.0):
  """Returns a `TimeStep` with `step_type` set to `StepType.LAST`."""
  return TimeStep(StepType.LAST, reward, discount, observations, avail_actions, state)


@dataclass
class Transition:
  observations:np.ndarray
  state:np.ndarray
  avail_actions:np.ndarray
  actions:np.ndarray
  rewards:np.ndarray
  terminated:bool

  def update(self, new):
    for key, value in new.items(): 
      if not hasattr(self, key): 
        self.__class__ = make_dataclass(
          self.__class__.__name__, 
          fields=[(key, type(value), field(init=False))], 
          bases=(self.__class__,)
        )
      if not isinstance(value, np.ndarray): value = np.array(value)
      setattr(self, key, value)


class Episode:
  def __init__(self, size):
    self._storage = []
    self._maxsize = size
    self._terminated = False
  
  def __len__(self):
    return len(self._storage)
  
  def __getitem__(self, item):
    return torch.cat([
      torch.tensor(np.expand_dims(getattr(timestep, item), axis=0), dtype=torch.float32) if not getattr(timestep, item).dtype == np.int64
      else torch.tensor(np.expand_dims(getattr(timestep, item), axis=0), dtype=torch.long) 
      for timestep in self._storage
    ], dim=0).unsqueeze(0)
  
  def __iter__(self):
    return iter(self._storage)

  @property
  def steps(self):
    return len(self._storage)

  @property
  def reward(self):
    return np.sum([t.rewards.squeeze() for t in self._storage])

  @property
  def terminated(self):
    return self._terminated
  
  def add(self, timestep):
    self._storage.append(timestep)
  
  def __repr__(self):
    return f"{type(self).__name__}(maxsize={self._maxsize}, size={len(self._storage)}, terminated={self._terminated})"


class EpisodeBatch:
  def __init__(self, batch_size: int, sequence_length: int, batch: dict):
    self._batch_size = batch_size
    self._sequence_length = sequence_length
    self._device = batch[list(batch.keys())[0]].device
    self._batch = batch
    self.keys = batch.keys
  
  @property
  def batch_size(self):
    return self._batch_size
  
  @property
  def sequence_length(self):
    return self._sequence_length

  @property
  def device(self):
    return self._device

  def __getitem__(self, item):
    assert item in self._batch.keys(), f"Key {item} not found!"
    return self._batch[item]


class ReplayBuffer:
  def __init__(self, size, device):
    """Replay Buffer.
    
    :param size (int): Max number of transitions to store in the buffer.
      When the buffer overflows the old memories are dropped.
    """
    self._storage = []
    self._maxsize = size
    self._next_idx = 0
    self._device = device
  
  def __len__(self):
    return len(self._storage)
  
  def add(self, data):
    if self._next_idx >= len(self._storage):
      self._storage.extend(data) if isinstance(data, list) else self._storage.append(data)
    else:
      if isinstance(data, list):
        idxs = slice(self._next_idx, self._next_idx + len(data))
        self._storage[idxs] = data
      else:
        self._storage[self._next_idx] = data
    self._next_idx = (
      (self._next_idx + len(data)) % self._maxsize if isinstance(data, list) else
      (self._next_idx + 1) % self._maxsize
    )
  
  def _max_lengths(self, batch):
    items = set()
    batch_size = set() 
    for key, value in batch.items():
      batch_size.add(len(batch[key]))
      for item in value:
        items.add(item.shape[1])
    assert len(batch_size) == 1, f"BatchSizeError! Batch-size is not unique, expected size 1 got {len(batch_size)}."
    return batch_size.pop(), max(items)

  def _encode_sample(self, idxs):
    # Create the batch by concatenating the episodes :see Episode.__getitem__:
    batch = {}
    for idx in idxs:
      episode = self._storage[idx]
      for k in asdict(next(iter(episode))):
        if k not in batch: batch[k] = []
        item = episode[k]
        batch[k].append(item)
    
    # Pad batch, create mask for padded values and concatentate batched episodes
    # batch["terminated"] = batch.pop("step_type")
    batch_size, sequence_length = self._max_lengths(batch)
    batch.update({"__mask__": torch.zeros((batch_size, sequence_length, 1), device=self._device)})
    for key, value in batch.items():
      if key == "__mask__": continue
      for i, item in enumerate(value):
        size = sequence_length - item.shape[1]
        pad = (0, 0) * (item.ndim - 2) + (0, size, 0, 0)
        padded = F.pad(item, pad=pad, mode='constant')
        value[i] = padded 
        batch["__mask__"][i, :item.shape[1]] = 1
      batch[key] = torch.cat(value, dim=0).to(self._device)

    return EpisodeBatch(len(idxs), sequence_length, batch)

  def can_sample(self, batch_size):
    return batch_size <= len(self._storage)

  def sample(self, batch_size) -> EpisodeBatch:
    """Sample a batch of experiences.
    
    :params batch_size (int): How many transitions to sample.
    :returns:
    """
    idxs = torch.randperm(len(self._storage) - 1)[:batch_size]
    return self._encode_sample(idxs)
  
  def __repr__(self):
    steps = sum([_.steps for _ in self._storage])
    return f"{type(self).__name__}(maxsize={self._maxsize}, size={len(self._storage)}, steps={steps})"

  def memory_usage(self, sizes: Dict[str, Tuple], dtypes: Tuple, sequence_length: int = 1):
    # Check the memory size without initializing values
    assert sizes.keys() == Transition.__annotations__.keys()
    nbytes = [
      np.prod([self._maxsize, sequence_length, *shape]) * np.dtype(dtype).itemsize 
      for (_, shape), dtype in zip(sizes.items(), dtypes)]
    total = sum(nbytes) / (1024 ** 3)
    return total
