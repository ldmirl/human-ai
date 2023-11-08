from __future__ import annotations
from typing import Type

import numpy as np

from multigrid import MultiGridEnv, Grid, Actions
from multigrid import objects, constants
from multigrid.envs import register

class Prey(objects.Ball):
  def __init__(self, env: Type[PredatorPrey], color="orange"):
    super().__init__(color=color)
    self.env = env
    self.alive = True

  def __getattr__(self, name: str):
    if name.startswith('__'):
      raise AttributeError(
        f"attempted to get missing private attribute '{name}'")
    return getattr(self.env, name)

  def step(self):
    # "smart" prey
    next_pos = None
    if self.alive:
      # try 5 samples and check to see if that move doesn't lead into neighborhood of agents
      for _ in range(5): 
        m = np.random.choice(len(self._prey_move_probs), 1, p=self._prey_move_probs)[0]
        next_pos = tuple([p + v for p, v in zip(self.cur_pos, constants.DIR_TO_VEC[m])]) \
          if m != 4 else self.cur_pos # no-op
        if self.neighbourhood(*next_pos, view=2)[0] == 0:
          break
      next_pos = self.cur_pos if next_pos is None else next_pos # default is no-op
      try:
        if self.grid.get(*next_pos) is None and not self.agent_here(*next_pos):
          self.grid.set(*self.cur_pos, None)
          self.grid.set(*next_pos, self)
          self.cur_pos = next_pos
      except AssertionError:
        pass

  def neighbourhood(self, x, y, view=None):
    # check if agent is in neighbourhood
    xy, count = [], 0
    view = self.agent_view_size if view is None else view
    for j in range(-view, view + 1):
      for i in range(-view, view + 1):
        coord = (x + j, y + i)
        if self.in_grid(*coord) and self.agent_here(*coord):
          xy.append(coord)
          count += 1
    idx = [k for k, pos in enumerate(self.agent_pos) if pos in xy]
    return count, idx


class MoveableBox(objects.Box):
  def __init__(self, color="yellow", n_hits=1):
    super().__init__(color=color)
    self.n_hits = n_hits
    self.hits = [0 for _ in range(4)] # UP, DOWN, LEFT, RIGHT
  
  def can_pickup(self): return False

  def toggle(self, env, agent_id, fwd_pos):
    self.hits[env.agent_dir[agent_id]] += 1
    move = [i for i, hit in enumerate(self.hits) if hit >= self.n_hits]
    if len(move) >= 1:
      obj = env.grid.get(*fwd_pos)
      next_pos = np.array(fwd_pos) + constants.DIR_TO_VEC[move[0]]
      fwd_cell = env.grid.get(*next_pos)
      if fwd_cell is None or fwd_cell.can_overlap():
        env.grid.set(*fwd_pos, None)
        env.grid.set(*next_pos, obj)
        # Reset number of hits in that direction
        self.hits[move[0]] = 0
        return True
    return False


class PredatorPrey(MultiGridEnv):
  def __init__(
    self,
    size=12,
    agents_pos=None,
    n_agents=2,
    n_preys=1,
    prey_move_probs=(0.175, 0.175, 0.175, 0.175, 0.3),
    max_steps=100,
    penalty=-0.5,
    step_cost=-0.1,
    prey_capture_reward=5,
    agent_view_size=5,
    use_movable_box=False,
    n_boxes=2,
    **kwargs,
  ):
    self.agents_default_pos = agents_pos
    self.n_preys = n_preys
    self.preys = [None for _ in range(self.n_preys)]
    self._prey_move_probs = prey_move_probs
    self._prey_capture_reward = prey_capture_reward
    self._penalty = penalty
    self._step_cost = step_cost
    self._use_movable_box = use_movable_box
    self.n_boxes = n_boxes

    super().__init__(
      grid_size=size,
      n_agents=n_agents,
      max_steps=max_steps,
      agent_view_size=agent_view_size,
      **kwargs,
    )
  
  def _generate_grid(self, width, height):
    # Create the grid
    self.grid = Grid(width, height)
    # Generate the surrounding walls
    self.grid.wall_rect(0, 0, width, height)
    # Randomize the player start position and orientation
    # TODO: deal with ``agents_default_pos``.
    self.place_agents()
    # Place preys on the grid. Specific to this environment.
    for p in range(self.n_preys):
      self.preys[p] = Prey(self)
      self.place_obj(self.preys[p])
    if self._use_movable_box:
      for _ in range(self.n_boxes):
        self.place_obj(MoveableBox(), max_tries=100)
  
  def step(self, actions):
    obs, rewards, terminated, info = super().step(actions)
    rewards = [r + self._step_cost for r in rewards]
    for p in range(self.n_preys):
      prey: Prey = self.preys[p]
      if prey.alive:
        # Check if agents have caught prey
        count = prey.neighbourhood(*prey.cur_pos, view=1)[0]
        if count >= 1:
          r = self._penalty if count == 1 else self._prey_capture_reward
          if count == 2:
            prey.alive = False
            self.grid.set(*prey.cur_pos, None)
          for i in range(self.n_agents): rewards[i] += r
        # Move the prey smartly if not caught
        prey.step()
    
    obs = self.gen_obs()
    if all([not self.preys[p].alive for p in range(self.n_preys)]):
      terminated = [True for _ in range(self.n_agents)]
    return obs, rewards, terminated, info


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

# for i, info in enumerate([(12, 2, 1, 2), (14, 4, 2, 4)]):
#   size, n_agents, n_preys, n_boxes = info
#   name = f"PredatorPrey"
#   register(
#     env_id=f"{name}-v{i}",
#     entry_point=f"{module_path}:PredatorPrey",
#     size=size,
#     n_agents=n_agents,
#     n_preys=n_preys,
#   )
#   register(
#     env_id=f"{name}-NoPenalty-v{i}",
#     entry_point=f"{module_path}:PredatorPrey",
#     size=size,
#     n_agents=n_agents,
#     n_preys=n_preys,
#     penalty=0,
#   )
#   # Fully-obsserved environment (each agent sees the entire grid)
#   register(
#     env_id=f"{name}-FullyObserved-v{i}",
#     entry_point=f"{module_path}:PredatorPrey",
#     size=size,
#     n_agents=n_agents,
#     n_preys=n_preys,
#     fully_observed=True,
#   )
#   # Prey is initialized at a random position and does not move
#   register(
#     env_id=f"{name}-RandomStatic-v{i}",
#     entry_point=f"{module_path}:PredatorPrey",
#     size=size,
#     n_agents=n_agents,
#     n_preys=n_preys,
#     prey_move_probs=(0, 0, 0, 0, 1),
#   )
#   # Both fully-observed and prey initialized at a random position and does not move
#   register(
#     env_id=f"{name}-FullyObserved-RandomStatic-v{i}",
#     entry_point=f"{module_path}:PredatorPrey",
#     size=size,
#     n_agents=n_agents,
#     n_preys=n_preys,
#     fully_observed=True,
#     prey_move_probs=(0, 0, 0, 0, 1),
#   )
#   # Movable boxes are placed randomly on the grid and agents can move them
#   register(
#     env_id=f"{name}-Moveable-Box-v{i}",
#     entry_point=f"{module_path}:PredatorPrey",
#     size=size,
#     n_agents=n_agents,
#     n_preys=n_preys,
#     use_movable_box=True,
#     n_boxes=n_boxes,
#   )
