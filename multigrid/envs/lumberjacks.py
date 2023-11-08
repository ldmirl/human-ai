from __future__ import annotations
from typing import Type

import numpy as np

from multigrid import MultiGridEnv, Grid
from multigrid import objects, constants
from multigrid.envs import register

class LumberJack(MultiGridEnv):
  def __init__(
      self,
      size=12,
      n_agents=2,
      n_trees=12,
      max_steps=100,
      step_cost=-0.1,
      tree_cutdown_reward=10,
      agent_view_size=5,
      **kwargs,
  ):
    self.n_trees = n_trees
    self._tree_map = [None for _ in range(self.n_trees)]
    self._tree_cutdown_reward = tree_cutdown_reward
    self._step_cost = step_cost

    super().__init__(
      grid_size=size,
      n_agents=n_agents,
      # max_steps=max_steps,
      max_steps=200,
      agent_view_size=agent_view_size,
      **kwargs,
    )
  
  def _generate_grid(self, width, height):
    self.grid = Grid(width, height)
    self.grid.wall_rect(0, 0, width, height)
    self.place_agents()
    # Reset the tree map and place trees
    self._tree_map = [None for _ in range(self.n_trees)]
    for tree in range(self.n_trees):
      stength = np.random.randint(1, self.n_agents + 1)
      self._tree_map[tree] = objects.Tree(stength)
      self.place_obj(self._tree_map[tree])
  
  def step(self, actions):
    obs, rewards, terminated, info = super().step(actions)
    rewards = [r + self._step_cost for r in rewards]
    for t, tree in enumerate(self._tree_map):
      nb = self._neighbourhood(*tree.cur_pos, view=1)
      if (
        nb[0] >= tree.strength and 
        all(x in tree.chops for x in nb[1]) and
        sum([1 for nt in tree.chops if nt in nb[1]]) >= tree.strength
      ):
        rewards = [r + self._tree_cutdown_reward if i in nb[1] else r for i, r in enumerate(rewards)]
        self.grid.set(*tree.cur_pos, None)
        self._tree_map.pop(t)
      # Make sure both agents choose to chop
      # the tree together
      try: self._tree_map[t].chops = set()
      except IndexError: pass
    obs = self.gen_obs()
    return obs, rewards, terminated, info

  # TODO: This is the same implementation as in Predator-Prey
  # Should probably move it as a method in Grid class.
  def _neighbourhood(self, x, y, view=None):
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
  
if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

# register(
#   env_id=f"LumberJack-v0",
#   entry_point=f"{module_path}:LumberJack",
# )
