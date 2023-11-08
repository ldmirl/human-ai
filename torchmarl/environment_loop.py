import numpy as np

from torchmarl.buffers import Episode, Transition
from torchmarl.buffers import restart, transit, terminate

class EnvironmentLoop:
  def __init__(self, env, actor, should_update=True):
    self._env = env
    self._actor = actor
    self._should_update = should_update

  def run_episode(self, num_episodes=0, num_steps=0, test=False):
    self._actor.init_hidden()
    # observation, state = self._env.reset()
    observation = self._env.reset()["image"]
    observation = observation.reshape(observation.shape[0], -1)
    state = observation
    avail_actions = np.ones((self._env.n_agents, self._env.n_actions))
    # Store previous action
    # prev_action = np.zeros((self._env.n_agents, self._env.n_actions))
    prev_action = np.zeros((self._env.n_agents, self._env.n_actions))
    eye = np.eye(self._env.n_agents)
    # observation = np.concatenate([observation, prev_action, eye], axis=1, dtype=np.float32)
    observation = np.concatenate([observation, prev_action, eye], axis=1, dtype=np.float32)
    # timestep = restart(np.expand_dims(observation, axis=0), np.expand_dims(self._env.get_avail_actions(), axis=0), state)
    timestep = restart(np.expand_dims(observation, axis=0), np.expand_dims(avail_actions, axis=0), state) 

    episode = Episode(size=self._env.max_steps)
    while not timestep.last():
      actions = self._actor.select_actions(timestep, num_steps, test=test)
      cpuactions = actions.squeeze().cpu().numpy()
      # reward, terminated, info = self._env.step(cpuactions)
      observation, rewards, terminated, _ = self._env.step(cpuactions)
      transition = Transition(
        observations=timestep.observation,
        state=timestep.state,
        avail_actions=timestep.avail_actions.squeeze(),
        actions=np.expand_dims(actions.squeeze().cpu().numpy(), axis=-1),
        rewards=np.array([np.sum(rewards)]),
        terminated=np.array([int(all(terminated))])
      )
      episode.add(transition)
      
      observation = observation["image"]
      # observation, state = self._env.get_obs(), self._env.get_state()
      observation = observation.reshape(observation.shape[0], -1)
      state = observation
      # avail_actions = np.expand_dims(self._env.get_avail_actions(), axis=0) 
      # Set action to one-hot
      # prev_action = np.zeros((self._env.n_agents, self._env.n_actions)) 
      prev_action = np.zeros((self._env.n_agents, self._env.n_actions))
      prev_action[np.arange(self._env.n_agents), cpuactions] = 1.
      observation = np.concatenate([observation, prev_action, eye], axis=1)
      observation = np.expand_dims(observation, axis=0)
      timestep = (
        transit(rewards, observation, np.expand_dims(avail_actions, axis=0), state)
        if not all(terminated) else terminate(rewards, observation, np.expand_dims(avail_actions, axis=0), state)
      )
    
    metrics = None
    if not test: 
      self._actor.buffer.add(episode)
      metrics = self._actor.update(num_episodes, num_steps)

    return episode, metrics, None 
