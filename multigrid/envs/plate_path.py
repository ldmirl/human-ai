import numpy as np

from multigrid import MultiGridEnv
from multigrid.grid import Grid
from multigrid.objects import Goal, Floor, Door
from multigrid.envs import register

class PlatePath(MultiGridEnv):
  def __init__(
    self, 
    size=9, 
    n_agents=2, 
    step_cost=-0.1,
    max_steps=100,
    **kwargs
  ):
    self.step_cost = step_cost

    super().__init__(
      grid_size=size,
      n_agents=n_agents,
      max_steps=max_steps,
      **kwargs,
    )

  def _generate_grid(self, width, height):
    self.grid = Grid(width, height)
    self.grid.wall_rect(0, 0, width, height)
    # Place a goal in the rightmost column at a random height
    random_height = self._rand_int(1, height - 2)
    self.put_obj(Goal(), width - 2, random_height)
    # Create a vertical splitting wall
    if width <= 5: start_idx = 2
    else: start_idx = 3
    self.split_idx = self._rand_int(start_idx, width - 3)
    self.grid.vert_wall(self.split_idx, 0)

    # Place a door in the wall
    door_idx = self._rand_int(1, width - 2)
    self.door_pos = (self.split_idx, door_idx)
    self.put_obj(Door('yellow', is_locked=True), self.split_idx, door_idx)

    self.place_agents(size=(self.split_idx, height))
    # Place a plate on the left side
    self.place_obj(obj=Floor(color="white"), top=(0, 0), size=(self.split_idx, height))
    # Place a plate on the right side
    self.place_obj(obj=Floor(color="white"), top=(self.split_idx, 0), size=(self.split_idx, height))
  
  def _reward(self) -> float:
    return super()._reward() + 10 
  
  def step(self, actions):
    obs, rewards, terminated, info = super().step(actions)
    rewards = [r + self.step_cost for r in rewards]
    cells = [self.grid.get(*agent.cur_pos) for agent in self.agent_obj]
    door = self.grid.get(*self.door_pos)
    cells = [cell is not None and cell.type == 'floor' for cell in cells]
    if any(cells):
      door.is_locked = False
      door.is_open = True
      self.grid.set(*self.door_pos, door)
      rewards[np.argmax(cells)] += -self.step_cost # Reward the agent that opened the door by removing step cost
    else:
      door.is_locked = True
      door.is_open = False
      self.grid.set(*self.door_pos, door)
    
    # Make sure all agents are in the rightmost quadrant
    if all(terminated):
      quadrant = [agent.cur_pos[0] >= self.split_idx for agent in self.agent_obj]
      if not all(quadrant):
        rewards = [self.step_cost - 1] * self.n_agents

    return obs, rewards, terminated, info

if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

# register(
#   env_id='PlatePath-v0',
#   entry_point=f'{module_path}:PlatePath',
# )
