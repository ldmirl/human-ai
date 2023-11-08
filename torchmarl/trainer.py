from typing import Mapping, Any
import collections
import pathlib
import os
import json
import numpy as np

from torchmarl.environment_loop import EnvironmentLoop

class Trainer:
  def __init__(
    self,
    env,
    actor,
    root_dir: pathlib.Path,
    *,
    num_steps: int = 10,
    num_episodes: int = None,
    test_episodes: int = 10,
    test_interval: int = 1000,
    log_interval: int = None,
    save_interval: int = 1000,
    use_wandb: bool = False,
    use_tqdm: bool = False,
    debug: bool = False,
  ):
    self.actor = actor
    self.env_loop = EnvironmentLoop(env, actor)
    self.root_dir = root_dir

    self.num_steps = num_steps
    self.num_episodes = num_episodes

    self.use_tqdm = use_tqdm

    self.test_episodes = test_episodes
    self.test_interval = test_interval
    self.prev_test_step = -test_interval
    self.save_interval = save_interval
    self.prev_save_step = 0
    self.debug = debug

    if not self.debug:
      # create root directory for saving checkpoints, logs, etc.
      os.makedirs(self.root_dir, exist_ok=True)
      folder = self.root_dir / f"{len(os.listdir(self.root_dir)) + 1}"
      for f in [folder, folder / "checkpoints"]:
        os.makedirs(f, exist_ok=True)
      self.folder = folder

  def should_terminate(self, n_steps, n_episodes):
    return (
      (self.num_steps > 0 and n_steps >= self.num_steps) or
      (self.num_episodes > 0 and n_episodes >= self.num_episodes)
    )

  def train(self):
    if self.use_tqdm: 
      from tqdm.auto import tqdm
      pbar = tqdm(total=self.num_steps, dynamic_ncols=True)
    
    num_steps, num_episodes = 0, 0
    while not self.should_terminate(num_steps, num_episodes):
      episode, info, metrics = self.env_loop.run_episode(num_episodes, num_steps)
      num_steps += len(episode)
      num_episodes += 1

      if num_steps - self.prev_test_step >= self.test_interval:
        if self.use_tqdm: pbar.set_description(f"testing...")
        test_rewards = self.test(num_steps, num_episodes)
        self.prev_test_step = num_steps

      if num_steps - self.prev_save_step >= self.save_interval:
        self.save(num_steps, num_episodes)
        self.prev_save_step = num_steps

      if self.use_tqdm:
        pbar.update(len(episode))
        pbar.set_description(f"episode: {num_episodes} train reward: {episode.reward:.4f} test reward: {np.mean(test_rewards):.4f}")
  
  def test(self, num_steps, num_episodes):
    test_rewards = []
    for _ in range(self.test_episodes):
      episode, info, _ = self.env_loop.run_episode(num_steps, num_episodes, test=True)
      test_rewards.append(episode.reward)
    return test_rewards
  
  def load(self):
    pass
  
  def save(self, num_steps, num_episodes):
    path = self.folder / "checkpoints"
    self.actor.save(path, num_episodes, num_steps)

  def backup(self, obj: dict, file: str):
    # Save the contents of ``obj`` for good measure.
    def _unposix(obj: Mapping[str, Any]) -> Mapping[str, Any]:
      for k, v in obj.items():
        if isinstance(v, pathlib.Path): obj[k] = str(v)
        elif isinstance(v, collections.abc.Mapping): obj[k] = _unposix(v)
        elif file != "config.json": # Unless the given file is config don't convert items to list
          if isinstance(v, int): obj[k] = [v] # Convert to list for JSON.
          elif k == "episode_return": obj[k] = [np.mean(v).item()]
      return obj
    
    path = self.folder / file
    pathlib.Path.touch(path, exist_ok=True)
    tmp = open(f"{path}.tmp", "w")
    with open(path, "r+") as f:
      data = json.loads(f.read()) if os.path.getsize(path) != 0 else {}
      data.update(_unposix(obj))
      json.dump(data, tmp, indent=2)
      os.rename(path, f"{path}.bak") # Backup old file.
      os.rename(f"{path}.tmp", path) # Rename new file.
      os.remove(f"{path}.bak") # Remove old file.
    return True