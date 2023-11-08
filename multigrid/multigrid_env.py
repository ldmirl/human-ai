from typing import Iterable, TypeVar, List
import abc
import math
from enum import IntEnum
from copy import copy

import numpy as np
import pygame

from multigrid.constants import (
  TILE_PIXELS,
  COLOR_NAMES,
  DIR_TO_VEC
)
from multigrid.grid import Grid
from multigrid.objects import Point, GridObject, Agent

T = TypeVar("T")

class Actions(IntEnum):
  # Turn left, turn right, move forward
  LEFT = 0
  RIGHT = 1
  FORWARD = 2
  # Pick up an object
  PICKUP = 3
  # Drop an object
  DROP = 4
  # Toggle/activate an object
  TOGGLE = 5
  # Done completing task
  DONE = 6


class MultiGridEnv:
  """2D grid world game environment with multi-agent support."""
  def __init__(
    self,
    grid_size: int | None = None,
    width: int | None = None,
    height: int | None = None,
    max_steps: int = 100,
    see_through_walls: bool = False,
    agent_view_size: int = 7,
    n_agents: int = 1,
    competitive: bool = False,
    fully_observed: bool = False,
    render_mode: str | None = "human",
    screen_size: int | None = 640,
    highlight: bool = True,
    tile_size: int = TILE_PIXELS,
    agent_pov: bool = False,
    seed: int | None = None,
  ) -> None:
    # Can't set both grid_size and width/height
    if grid_size:
      assert width is None and height is None
      width = height = grid_size
    assert width is not None and height is not None

    # Number of cells (width and height) in the agent view
    assert agent_view_size % 2 == 1 and agent_view_size >= 3
    self.agent_view_size = agent_view_size
    # if fully_observed: self.agent_view_size = max(width, height)

    # Range of possible rewards
    self.reward_range = (0, 1)

    # Window to use for human rendering mode
    self.screen_size = screen_size
    self.render_size = None
    self.window = None
    self.clock = None

    # Environment configuration
    self.width, self.height = width, height
    assert isinstance(
        max_steps, int
    ), f"The argument max_steps must be an integer, got: {type(max_steps)}"
    self.max_steps = max_steps
    self.see_through_walls = see_through_walls

    # Set the number of agents
    self.n_agents = n_agents
    # Set the number of actions
    self.n_actions = len(Actions)
    # If competitive, only one agent is allowed to reach the goal.
    self.competitive = competitive or self.n_agents == 1
    # Current position and direction of the agent
    self.agent_pos = [None] * self.n_agents
    self.agent_dir = [None] * self.n_agents
    # Maintain a done variable for each agent
    self.agent_done = [False] * self.n_agents

    # TODO: Convert above to agent objects internals
    # Remove agent objects from grid...they are causing issues with rendering overlaps...
    self.agent_obj: List[Agent | None] = [None] * self.n_agents

    # Current grid and mission and carrying
    self.grid = Grid(width, height)
    self.carrying = None

    # Rendering attributes
    self.render_mode = render_mode
    self.highlight = highlight
    self.tile_size = tile_size
    self.agent_pov = agent_pov
    self.fully_observed = fully_observed
    self.render_fps = 30

    self.step_count = None

    # Initialize the RNG
    self._rng = np.random.RandomState(seed)

  @abc.abstractmethod
  def _generate_grid(self, width, height):
    pass

  def reset(self):
    # Current position and direction of the agent
    self.agent_pos = [None] * self.n_agents
    self.agent_dir = [None] * self.n_agents
    self.agent_done = [False] * self.n_agents

    # Generate a new random grid at the start of each episode
    self._generate_grid(self.width, self.height)

    for a in range(self.n_agents):
      assert self.agent_pos[a] is not None and self.agent_dir[a] is not None
      assert self.agent_pos[a] >= (0, 0) and self.agent_dir[a] >= 0
      # Check that the agent doesn't overlap with an object
      start_cell = self.grid.get(*self.agent_pos[a])
      assert start_cell is None or start_cell.can_overlap()

    # Item picked up, being carried, initially nothing
    self.carrying = [None] * self.n_agents
    # Step count since episode start
    self.step_count = 0

    # Return first observation
    obs = self.gen_obs()

    return obs

  @property
  def steps_remaining(self):
    return self.max_steps - self.step_count

  @property
  def dir_vec(self):
    """Get the direction vector for the agent, pointing in the direction of forward movement."""
    ret = []
    for a in range(self.n_agents):
      agent_dir = self.agent_dir[a]
      assert agent_dir >= 0 and agent_dir < 4, \
        f"Invalid agent_dir: {agent_dir} is not within range(0, 4)"
      ret.append(DIR_TO_VEC[agent_dir])
    return ret

  @property
  def right_vec(self):
    """Get the vector pointing to the right of the agent."""
    return [np.array((-dy, dx)) for dx, dy in self.dir_vec]

  @property
  def front_pos(self):
    """Get the position of the cell that is right in front of the agent"""
    front_pos = [None for _ in range(self.n_agents)]
    for a in range(self.n_agents):
      assert self.agent_pos[a] is not None and self.dir_vec[a] is not None
      front_pos[a] = self.agent_pos[a] + self.dir_vec[a]
    return front_pos

  def __str__(self):
    """Produce a pretty string of the environment's grid along with the agent.
    A grid cell is represented by 2-character string, the first one for
    the object and the second one for the color.
    """

    # Map of object types to short string
    OBJECT_TO_STR = {
        "wall": "W",
        "floor": "F",
        "door": "D",
        "key": "K",
        "ball": "A",
        "box": "B",
        "goal": "G",
        "lava": "V",
    }

    # Map agent's direction to short string
    AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

    output = ""

    for j in range(self.grid.height):
      for i in range(self.grid.width):
        if i == self.agent_pos[0] and j == self.agent_pos[1]:
          output += 2 * AGENT_DIR_TO_STR[self.agent_dir]
          continue

        tile = self.grid.get(i, j)

        if tile is None:
          output += "  "
          continue

        if tile.type == "door":
          if tile.is_open:
            output += "__"
          elif tile.is_locked:
            output += "L" + tile.color[0].upper()
          else:
            output += "D" + tile.color[0].upper()
          continue

        output += OBJECT_TO_STR[tile.type] + tile.color[0].upper()

      if j < self.grid.height - 1:
          output += "\n"

    return output


  def _reward(self) -> float:
    """Compute the reward to be given upon success"""
    return 1 - 0.9 * (self.step_count / self.max_steps)

  def _rand_int(self, low: int, high: int) -> int:
    """Generate random integer in [low,high["""
    return self._rng.randint(low, high)

  def _rand_float(self, low: float, high: float) -> float:
    """Generate random float in [low,high["""
    return self._rng.uniform(low, high)

  def _rand_bool(self) -> bool:
    """Generate random boolean value"""
    return self._rng.randint(0, 2) == 0

  def _rand_elem(self, iterable: Iterable[T]) -> T:
    """Pick a random element in a list"""
    lst = list(iterable)
    idx = self._rand_int(0, len(lst))
    return lst[idx]

  def _rand_subset(self, iterable: Iterable[T], num_elems: int) -> list[T]:
    """Sample a random subset of distinct elements of a list"""

    lst = list(iterable)
    assert num_elems <= len(lst)

    out: list[T] = []

    while len(out) < num_elems:
      elem = self._rand_elem(lst)
      lst.remove(elem)
      out.append(elem)

    return out

  def _rand_color(self) -> str:
    """Generate a random color name (string)"""
    return self._rand_elem(COLOR_NAMES)

  def _rand_pos(
      self, x_low: int, x_high: int, y_low: int, y_high: int
  ) -> tuple[int, int]:
    """Generate a random (x,y) position tuple"""
    return (
      self._rng.randint(x_low, x_high), self._rng.randint(y_low, y_high)
    )

  def place_obj(self, obj:GridObject, top:Point=None, size: tuple[int, int] = None, reject_fn=None, max_tries=math.inf):
    """Place an object at an empty position in the grid

    :param top: top-left position of the rectangle where to place
    :param size: size of the rectangle where to place
    :param reject_fn: function to filter out potential positions
    """
    top = (0, 0) if top is None else (max(top[0], 0), max(top[1], 0))
    if size is None: size = (self.grid.width, self.grid.height)

    num_tries = 0
    while True:
      # This is to handle with rare cases where rejection sampling
      # gets stuck in an infinite loop
      if num_tries > max_tries: raise RecursionError("rejection sampling failed in place_obj")
      num_tries += 1
      pos = (
        self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
        self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
      )

      # Don't place the object on top of another object
      if self.grid.get(*pos) is not None: continue
      # Don't place the object where the agent is
      if any([np.array_equal(pos, self.agent_pos[a]) for a in range(self.n_agents)]): continue
      # Check if there is a filtering criterion
      if reject_fn and reject_fn(self, pos): continue
      break

    self.grid.set(pos[0], pos[1], obj)

    if obj is not None:
      obj.init_pos = pos
      obj.cur_pos = pos

    return pos

  def put_obj(self, obj:GridObject, i:int, j:int):
    """Put an object at a specific position in the grid"""
    self.grid.set(i, j, obj)
    obj.init_pos = (i, j)
    obj.cur_pos = (i, j)

  def place_agents(self, top=None, size=None, rand_dir=True, max_tries=math.inf):
    """Set the agent's starting point at an empty position in the grid"""
    for a in range(self.n_agents):
      self.place_agent(a, top, size, rand_dir, max_tries)

  def place_agent(self, idx, top=None, size=None, rand_dir=True, max_tries=math.inf):
    self.agent_pos[idx] = (-1, -1)
    pos = self.place_obj(None, top, size, max_tries=max_tries)
    _, dir = self.set_agent_pos(idx, pos, rand_dir)
    return pos, dir

  def set_agent_pos(self, idx, pos, rand_dir=True):
    self.agent_pos[idx] = pos
    if rand_dir: self.agent_dir[idx] = self._rand_int(0, 4)

    # if not agent_obj: agent_obj = Agent(idx, self.agent_dir[idx])
    # else: agent_obj.dir = self.agent_dir[idx]
    agent_obj = Agent(idx, self.agent_dir[idx])
    agent_obj.init_pos = pos
    self.agent_obj[idx] = agent_obj

    agent_obj.cur_pos = pos
    # self.grid.set(pos[0], pos[1], agent_obj)
    return pos, self.agent_dir[idx]

  def get_view_coords(self, i, j):
      """
      Translate and rotate absolute grid coordinates (i, j) into the
      agent's partially observable view (sub-grid). Note that the resulting
      coordinates may be negative or outside of the agent's view size.
      """

      ax, ay = self.agent_pos
      dx, dy = self.dir_vec
      rx, ry = self.right_vec

      # Compute the absolute coordinates of the top-left view corner
      sz = self.agent_view_size
      hs = self.agent_view_size // 2
      tx = ax + (dx * (sz - 1)) - (rx * hs)
      ty = ay + (dy * (sz - 1)) - (ry * hs)

      lx = i - tx
      ly = j - ty

      # Project the coordinates of the object relative to the top-left
      # corner onto the agent's own coordinate system
      vx = rx * lx + ry * ly
      vy = -(dx * lx + dy * ly)

      return vx, vy

  def get_view_exts(self, idx=None, agent_view_size=None):
      """
      Get the extents of the square set of tiles visible to the agent
      Note: the bottom extent indices are not included in the set
      if agent_view_size is None, use self.agent_view_size
      """

      agent_view_size = agent_view_size or self.agent_view_size
      agent_pos, agent_dir = self.agent_pos[idx], self.agent_dir[idx]

      # Facing right
      if agent_dir == 0:
          topX = agent_pos[0]
          topY = agent_pos[1] - agent_view_size // 2
      # Facing down
      elif agent_dir == 1:
          topX = agent_pos[0] - agent_view_size // 2
          topY = agent_pos[1]
      # Facing left
      elif agent_dir == 2:
          topX = agent_pos[0] - agent_view_size + 1
          topY = agent_pos[1] - agent_view_size // 2
      # Facing up
      elif agent_dir == 3:
          topX = agent_pos[0] - agent_view_size // 2
          topY = agent_pos[1] - agent_view_size + 1
      else:
          assert False, "invalid agent direction"

      botX = topX + agent_view_size
      botY = topY + agent_view_size

      return topX, topY, botX, botY

  def relative_coords(self, x, y):
      """
      Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
      """

      vx, vy = self.get_view_coords(x, y)

      if vx < 0 or vy < 0 or vx >= self.agent_view_size or vy >= self.agent_view_size:
          return None

      return vx, vy

  def in_view(self, x, y):
      """
      check if a grid position is visible to the agent
      """

      return self.relative_coords(x, y) is not None

  def in_grid(self, x, y):
      """Check if a position is within the grid bounds."""
      return x >= 0 and x < self.width and y >= 0 and y < self.height

  def agent_here(self, x, y):
    """Check if an agent is at a given position."""
    for pos  in self.agent_pos:
      if pos == (x, y): return True
    return False

  def agent_sees(self, x, y):
      """
      Check if a non-empty grid position is visible to the agent
      """

      coordinates = self.relative_coords(x, y)
      if coordinates is None:
          return False
      vx, vy = coordinates

      obs = self._observation()

      obs_grid, _ = Grid.decode(obs["image"])
      obs_cell = obs_grid.get(vx, vy)
      world_cell = self.grid.get(x, y)

      assert world_cell is not None

      return obs_cell is not None and obs_cell.type == world_cell.type

  def _forward(self, agent_id, fwd_pos, agent_obj:Agent):
    """Attempts to move the forward one cell, returns True if successful."""
    fwd_cell = self.grid.get(*fwd_pos)
    # Make sure agents can't walk into each other
    agent_blocking = False
    for a in range(self.n_agents):
      if a != agent_id and np.array_equal(self.agent_pos[a], fwd_pos):
        agent_blocking = True

    # Deal with object interactions
    # TODO: complete each condition in respect to the environment
    if not agent_blocking:
      if fwd_cell is not None and fwd_cell.type == "goal":
        return True, fwd_cell
      elif fwd_cell is not None and fwd_cell.type == "lava":
        pass
      elif fwd_cell is None or fwd_cell.can_overlap():
        # self.grid.set(*self.agent_pos[agent_id], None)
        # self.grid.set(*fwd_pos, agent_obj)
        self.agent_pos[agent_id] = tuple(fwd_pos)
        return True, fwd_cell
    return False, fwd_cell

  def _step(self, agent_id, action):
    reward = 0
    terminated = False
    # Get the position in front of the agent
    fwd_pos = self.front_pos[agent_id]
    # agent_obj: Agent = self.grid.get(*self.agent_pos[agent_id])
    agent_obj = self.agent_obj[agent_id]

    # Rotate left
    if action == Actions.LEFT:
      self.agent_dir[agent_id] -= 1
      if self.agent_dir[agent_id] < 0:
        self.agent_dir[agent_id] += 4
      agent_obj.agent_dir = self.agent_dir[agent_id]

    # Rotate right
    elif action == Actions.RIGHT:
      self.agent_dir[agent_id] = (self.agent_dir[agent_id] + 1) % 4
      agent_obj.agent_dir = self.agent_dir[agent_id]

    # Move forward
    elif action == Actions.FORWARD:
      successful_forward, fwd_cell = self._forward(agent_id, fwd_pos, agent_obj)
      if successful_forward:
        obj = agent_obj if not isinstance(agent_obj, tuple) else agent_obj[-1]
        obj.cur_pos = fwd_pos
        if fwd_cell is not None and fwd_cell.type == "goal":
          terminated = True
          reward = self._reward()

    # Pick up an object
    elif action == Actions.PICKUP:
      fwd_cell = self.grid.get(*fwd_pos)
      if fwd_cell and fwd_cell.can_pickup():
        if self.carrying[agent_id] is None:
          self.carrying[agent_id] = fwd_cell
          self.carrying[agent_id].cur_pos = np.array([-1, -1])
          self.grid.set(*fwd_pos, None)
          agent_obj.contains = fwd_cell

    # Drop an object
    elif action == Actions.DROP:
      fwd_cell = self.grid.get(*fwd_pos)
      if not fwd_cell and self.carrying[agent_id]:
        self.grid.set(*fwd_pos, self.carrying[agent_id])
        self.carrying[agent_id].cur_pos = fwd_pos
        self.carrying[agent_id] = None
        agent_obj.contains = None

    # Toggle/activate an object
    elif action == Actions.TOGGLE:
      fwd_cell = self.grid.get(*fwd_pos)
      if fwd_cell:
        fwd_cell.toggle(self, agent_id, fwd_pos)

    # Done action (not used by default)
    elif action == Actions.DONE:
      pass
    else:
      raise ValueError(f"Expected action in range (0, {len(Actions) - 1}) but received: {action}")

    return reward, terminated

  def step(self, actions):
    assert self.step_count is not None, "Call reset before using step."
    self.step_count += 1
    rewards = [0 for _ in range(self.n_agents)]
    terminated = [False for _ in range(self.n_agents)]

    # Randomize order in which agents act for fairness
    agent_order = np.arange(self.n_agents)
    self._rng.shuffle(agent_order)

    # Step each agent
    for a in agent_order:
      rewards[a], terminated[a] = self._step(a, actions[a])

    obs = self.gen_obs()
    terminated = [any(terminated) for _ in range(self.n_agents)]
    # Running out of time applies to all agents
    if self.step_count >= self.max_steps:
      terminated = [True for _ in range(self.n_agents)]

    return obs, rewards, terminated, {}

  def gen_obs_grid(self, idx, agent_view_size=None):
    """
    Generate the sub-grid observed by the agent.
    This method also outputs a visibility mask telling us which grid
    cells the agent can actually see.
    if agent_view_size is None, self.agent_view_size is used
    """
    topX, topY, _, _ = self.get_view_exts(idx, agent_view_size)
    agent_view_size = agent_view_size or self.agent_view_size
    # print(self.grid, topX, topY, agent_view_size)
    grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

    for _ in range(self.agent_dir[idx] + 1):
      grid = grid.rotate_left()

    # Process occluders and visibility
    # Note that this incurs some performance cost
    if not self.see_through_walls:
      vis_mask = grid.process_vis(
        agent_pos=(agent_view_size // 2, agent_view_size - 1)
      )
    else:
      vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

    # Make it so the agent sees what it's carrying
    # We do this by placing the carried object at the agent's position
    # in the agent's partially observable view
    agent_pos = grid.width // 2, grid.height - 1
    if self.carrying[idx]:
      grid.set(*agent_pos, self.carrying[idx])
    else:
      grid.set(*agent_pos, None)

    return grid, vis_mask

  def gen_obs(self):
    """Generate the agent's view (partially observable, low-resolution encoding)"""
    obs = []
    for a in range(self.n_agents):
      grid, vis_mask = self.gen_obs_grid(a)
      # Encode the partially observable view into a numpy array
      img = grid.encode(vis_mask) if not self.fully_observed else self.grid.encode()
      img = np.expand_dims(img, axis=0)
      obs.append(img)

    image = np.concatenate(obs, axis=0)

    # Observations are dictionaries containing:
    # - an image (partially observable view of the environment)
    # - the agent's direction/orientation (acting as a compass)
    # - the agent's position (only if fully observable)
    obs = {"image": image, "direction": copy(self.agent_dir)}
    if self.fully_observed:
      obs["position"] = copy(self.agent_pos)

    return obs

  def get_pov_render(self, tile_size):
      """
      Render an agent's POV observation for visualization
      """
      grid, vis_mask = self.gen_obs_grid(0)
      print(grid.grid)
      # Render the whole grid
      img = grid.render(
          tile_size,
          # agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
          # agent_dir=3,
          agent_pos=None,
          agent_dir=None,
          highlight_mask=vis_mask,
      )

      return img

  def get_full_render(self, highlight, tile_size):
    """Render a non-paratial observation for visualization"""

    # Mask of which cells to highlight
    highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)
    # for idx in range(self.n_agents):
    idx = 0
    # Compute which cells are visible to the agent
    _, vis_mask = self.gen_obs_grid(idx)

    # Compute the world coordinates of the bottom-left corner
    # of the agent's view area
    f_vec = self.dir_vec[idx]
    r_vec = self.right_vec[idx]
    top_left = (
      self.agent_pos[idx]
      + f_vec * (self.agent_view_size - 1)
      - r_vec * (self.agent_view_size // 2)
    )

    # For each cell in the visibility mask
    for vis_j in range(0, self.agent_view_size):
      for vis_i in range(0, self.agent_view_size):
        # If this cell is not visible, don't highlight it
        if not vis_mask[vis_i, vis_j]: continue
        # Compute the world coordinates of this cell
        abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
        # Skip this cell if it's outside the grid bounds
        if abs_i < 0 or abs_i >= self.width: continue
        if abs_j < 0 or abs_j >= self.height: continue
        # Mark this cell to be highlighted
        highlight_mask[abs_i, abs_j] = True

    # Render the whole grid
    img = self.grid.render(
        tile_size,
        agents=self.agent_obj,
        highlight_mask=highlight_mask if highlight else None,
    )
    return img

  def get_frame(self, highlight:bool=True, tile_size:int=TILE_PIXELS, agent_pov:bool=False):
    """Returns an RGB image corresponding to the whole environment or the agent's point of view.

    Args:
      highlight (bool): If true, the agent's field of view or point of view is highlighted with a lighter gray color.
      tile_size (int): How many pixels will form a tile from the NxM grid.
      agent_pov (bool): If true, the rendered frame will only contain the point of view of the agent.

    Returns:
      frame (np.ndarray): A frame of type numpy.ndarray with shape (x, y, 3) representing RGB values for the x-by-y pixel image.
    """
    if agent_pov: return self.get_pov_render(tile_size)
    return self.get_full_render(highlight, tile_size)

  def render(self):
    img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)
    if self.render_mode == "human":
      img = np.transpose(img, axes=(1, 0, 2))
      if self.render_size is None:
        self.render_size = img.shape[:2]
      if self.window is None:
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode(
          (self.screen_size, self.screen_size)
        )
        pygame.display.set_caption("multigrid")
      if self.clock is None:
        self.clock = pygame.time.Clock()
      surf = pygame.surfarray.make_surface(img)

      # Create background with mission description
      offset = surf.get_size()[0] * 0.025
      # offset = 32 if self.agent_pov else 64
      bg = pygame.Surface(
        (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
      )
      bg.convert()
      bg.fill((255, 255, 255))
      bg.blit(surf, (offset / 2, offset / 2))

      bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

      self.window.blit(bg, (0, 0))
      pygame.event.pump()
      self.clock.tick(self.render_fps)
      pygame.display.flip()

    elif self.render_mode == "rgb_array":
      return img

  def close(self):
    if self.window: pygame.quit()
