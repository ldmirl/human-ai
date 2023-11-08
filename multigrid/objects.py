from __future__ import annotations
from functools import partial
from typing import Tuple
import math

import numpy as np

from multigrid import rendering
from multigrid.constants import (
    IDX_TO_OBJECT,
    IDX_TO_COLOR,
    OBJECT_TO_IDX,
    COLOR_TO_IDX,
    COLORS,
    AGENT_COLOURS
)

Point = Tuple[int, int]


class GridObject:
  """Base class for grid-world objects"""
  def __init__(self, type:str, color:str=None):
    assert type in OBJECT_TO_IDX, type
    self.type = type
    if color:
      self.color = color
    self.contains = None
    # Initial position of the object
    self.init_pos: Point | None = None
    # Current position of the object
    self.cur_pos: Point | None = None

  """Can the agent overlap with this?"""
  def can_overlap(self) -> bool: return False
  """Can the agent pick this up?"""
  def can_pickup(self) -> bool: return False
  """Can this contain another object?"""
  def can_contain(self) -> bool: return False
  """Can the agent see behind this object?"""
  def see_behind(self) -> bool: return True
  """Method to trigger/toggle an action this object performs"""
  def toggle(self, env, agent_id:int, pos:Point) -> bool: return False

  def encode(self) -> tuple[int, int, int]:
      """Encode the a description of this object as a 3-tuple of integers"""
      return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

  @staticmethod
  def decode(type_idx: int, color_idx: int, state: int) -> GridObject | None:
    """Create an object from a 3-tuple state description"""
    obj_type, color = IDX_TO_OBJECT[type_idx], IDX_TO_COLOR[color_idx]
    if obj_type == "empty" or obj_type == "unseen": return None

    _type = {
      "wall": partial(Wall, color),
      "floor": partial(Floor, color),
      "ball": partial(Ball, color),
      "key": partial(Key, color),
      "box": partial(Box, color),
      "door": partial(Door, color, state == 0, state == 2), # State, 0: open, 1: closed, 2: locked
      "goal": partial(Goal),
      "lava": partial(Lava),
      "tree": partial(Tree),
      "agent": partial(Agent, color_idx, state),
    }
    assert obj_type in _type, f"unknown object type in decode '{obj_type}'"
    return _type[obj_type]()

  def render(self, r: np.ndarray) -> np.ndarray:
      """Draw this object with the given renderer"""
      raise NotImplementedError

class Goal(GridObject):
  def __init__(self):
    super().__init__("goal", "green")

  def can_overlap(self): return True

  def render(self, img):
    rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Floor(GridObject):
  """Colored floor tile the agent can walk over"""
  def __init__(self, color: str = "blue"):
    super().__init__("floor", color)

  def can_overlap(self): return True

  def render(self, img):
    # Give the floor a pale color
    color = COLORS[self.color] / 2
    rendering.fill_coords(img, rendering.point_in_rect(0.031, 1, 0.031, 1), color)


class Lava(GridObject):
  def __init__(self):
    super().__init__("lava", "red")

  def can_overlap(self): return True

  def render(self, img):
    c = (255, 128, 0)
    # Background color
    rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), c)
    # Little waves
    for i in range(3):
      ylo = 0.3 + 0.2 * i
      yhi = 0.4 + 0.2 * i
      rendering.fill_coords(img, rendering.point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
      rendering.fill_coords(img, rendering.point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
      rendering.fill_coords(img, rendering.point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
      rendering.fill_coords(img, rendering.point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Wall(GridObject):
  def __init__(self, color: str = "grey"):
    super().__init__("wall", color)

  def see_behind(self):
    return False

  def render(self, img):
    rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Door(GridObject):
  def __init__(self, color: str, is_open: bool = False, is_locked: bool = False):
    super().__init__("door", color)
    self.is_open = is_open
    self.is_locked = is_locked

  """The agent can only walk over this cell when the door is open"""
  def can_overlap(self): return self.is_open
  def see_behind(self): return self.is_open

  def toggle(self, env, agent_id, pos):
    # If the player has the right key to open the door
    if self.is_locked:
      if isinstance(env.carrying[agent_id], Key) and env.carrying[agent_id].color == self.color:
        self.is_locked = False
        self.is_open = True
        return True
      return False

    self.is_open = not self.is_open
    return True

  def encode(self):
    """Encode the a description of this object as a 3-tuple of integers"""

    # State, 0: open, 1: closed, 2: locked
    if self.is_open: state = 0
    elif self.is_locked: state = 2
    # if door is closed and unlocked
    elif not self.is_open: state = 1
    else:
      raise ValueError(
        "There is no possible state encoding for the state:\n"
        f"-Door Open: {self.is_open}\n"
        f"-Door Closed: {not self.is_open}\n"
        f"-Door Locked: {self.is_locked}"
      )

    return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

  def render(self, img):
    c = COLORS[self.color]

    if self.is_open:
      rendering.fill_coords(img, rendering.point_in_rect(0.88, 1.00, 0.00, 1.00), c)
      rendering.fill_coords(img, rendering.point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
      return

    # Door frame and door
    if self.is_locked:
      rendering.fill_coords(img, rendering.point_in_rect(0.00, 1.00, 0.00, 1.00), c)
      rendering.fill_coords(img, rendering.point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

      # Draw key slot
      rendering.fill_coords(img, rendering.point_in_rect(0.52, 0.75, 0.50, 0.56), c)
    else:
      rendering.fill_coords(img, rendering.point_in_rect(0.00, 1.00, 0.00, 1.00), c)
      rendering.fill_coords(img, rendering.point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
      rendering.fill_coords(img, rendering.point_in_rect(0.08, 0.92, 0.08, 0.92), c)
      rendering.fill_coords(img, rendering.point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

      # Draw door handle
      rendering.fill_coords(img, rendering.point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class Key(GridObject):
  def __init__(self, color: str = "blue"):
    super().__init__("key", color)

  def can_pickup(self): return True

  def render(self, img):
    c = COLORS[self.color]

    # Vertical quad
    rendering.fill_coords(img, rendering.point_in_rect(0.50, 0.63, 0.31, 0.88), c)

    # Teeth
    rendering.fill_coords(img, rendering.point_in_rect(0.38, 0.50, 0.59, 0.66), c)
    rendering.fill_coords(img, rendering.point_in_rect(0.38, 0.50, 0.81, 0.88), c)

    # Ring
    rendering.fill_coords(img, rendering.point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
    rendering.fill_coords(img, rendering.point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Ball(GridObject):
  def __init__(self, color="blue"):
    super().__init__("ball", color)

  def can_pickup(self): return True

  def render(self, img):
    rendering.fill_coords(img, rendering.point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Box(GridObject):
  def __init__(self, color, contains: GridObject | None = None):
    super().__init__("box", color)
    self.contains = contains

  def can_pickup(self): return True

  def toggle(self, env, agent_id, pos):
    # Replace the box by its contents
    env.grid.set(pos[0], pos[1], self.contains)
    return True

  def render(self, img):
    c = COLORS[self.color]

    # Outline
    rect_fn = rendering.point_in_rect(0.12, 0.88, 0.12, 0.88)
    rendering.fill_coords(img, rect_fn, c)
    rect_fn = rendering.point_in_rect(0.18, 0.82, 0.18, 0.82)
    rendering.fill_coords(img, rect_fn, (0, 0, 0))

    # Horizontal slit
    rect_fn = rendering.point_in_rect(0.16, 0.84, 0.47, 0.53)
    rendering.fill_coords(img, rect_fn, c)


class Tree(GridObject):
  def __init__(self, strength=1, color="green"):
    super().__init__(type="tree", color=color)
    # The number of agents required to cut down this tree
    self.strength = strength
    self.chops = set()

  def toggle(self, env, agent_id, pos):
    self.chops.add(agent_id)
    return True

  def encode(self) -> tuple[int, int, int]:
    """Encode the a description of this object as a 3-tuple of integers"""
    return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.strength)

  def render(self, img):
    # Tree leaves
    rendering.fill_coords(img, rendering.point_in_rect(0.1, 0.35, 0.1, 0.4), np.array([141, 203, 105]))
    rendering.fill_coords(img, rendering.point_in_rect(0.35, 0.65, 0.1, 0.4), np.array([122, 175, 90]))
    rendering.fill_coords(img, rendering.point_in_rect(0.65, 0.9, 0.1, 0.4), np.array([123, 194, 80]))
    rendering.fill_coords(img, rendering.point_in_rect(0.1, 0.35, 0.4, 0.7), np.array([161, 217, 127]))
    rendering.fill_coords(img, rendering.point_in_rect(0.65, 0.9, 0.4, 0.7), np.array([102, 162, 67]))
    # Tree trunk
    rendering.fill_coords(img, rendering.point_in_rect(0.35, 0.65, 0.4, 0.7), np.array([167, 106, 43]))
    rendering.fill_coords(img, rendering.point_in_rect(0.35, 0.65, 0.7, 0.9), np.array([189, 128, 65]))


class Circle(GridObject):
  def __init__(self, color:str="purple"):
    super().__init__("circle", color)

  def render(self, img):
    circle_fn = rendering.point_in_circle(0.5, 0.5, 0.31)
    color = COLORS[self.color]
    rendering.fill_coords(img, circle_fn, color)


class Agent(GridObject):
  """Class to represent other agents existing in the world."""
  def __init__(self, agent_id, agent_dir):
    super(Agent, self).__init__("agent")
    self.agent_id = agent_id
    self.agent_dir = agent_dir
    self.agent_done = False

  """Can this contain another object?"""
  def can_contain(self): return True

  def encode(self):
    """Encode the a description of this object as a 3-tuple of integers."""
    return (OBJECT_TO_IDX[self.type], self.agent_id, self.agent_dir)

  def render(self, img):
    tri_fn = rendering.point_in_triangle(
        (0.12, 0.19),
        (0.87, 0.50),
        (0.12, 0.81),
    )

    # Rotate the agent based on its direction
    tri_fn = rendering.rotate_fn(
        tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.agent_dir)
    color = AGENT_COLOURS[self.agent_id]
    rendering.fill_coords(img, tri_fn, color)
