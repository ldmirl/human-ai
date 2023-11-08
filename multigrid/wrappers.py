import abc
from einops import rearrange
import dm_env

class Wrapper:
  def __init__(self, env):
    self.env = env

  def __getattr__(self, name: str):
    if name.startswith("_"):
      raise AttributeError(f"accessing private attribute '{name}' is prohibited")
    return getattr(self.env, name)
  
  def __str__(self):
    """Returns the wrapper name and the :attr:`env` representation string."""
    return f"<{type(self).__name__}{self.env}>"

  def reset(self):
    """Uses the :meth:`reset` of the :attr:`env` that can be overwritten to change the returned data."""
    return self.env.reset()
  
  def step(self, actions):
    """Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data."""
    return self.env.step(actions)
  
  def close(self):
    """Closes the wrapper and :attr:`env`."""
    return self.env.close()
  
class ObservationWrapper(Wrapper):
  def __init__(self, env):
    super().__init__(env)

  def reset(self):
    obs = self.env.reset()
    return self.observation(obs)

  def step(self, actions):
    obs, rewards, terminated, info = self.env.step(actions)
    return self.observation(obs), rewards, terminated, info

  @abc.abstractmethod
  def observation(self, obs):
    """Returns a modified observation."""
    raise NotImplementedError

class DMEnvWrapper(Wrapper):
  def __init__(self, env):
    super().__init__(env)

  def reset(self):
    obs = self.env.reset()
    return dm_env.restart(obs)

  def step(self, actions):
    obs, rewards, terminated, info = self.env.step(actions)
    if all(terminated):
      return dm_env.termination(rewards, obs), info
    return dm_env.transition(rewards, obs), info

class FlatObsWrapper(ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)

  def observation(self, obs):
    image = obs["image"]
    obs = rearrange(image, "n h w c -> n (h w c)")
    return obs
