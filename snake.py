from config import FOOD_COLOR, WHITE, SNAKE_COLOR

NUM_LABELS = 4 # blank, food, wall, snake
NUM_ACTIONS = 4

class Label:
  BLANK = 0
  FOOD = 1
  WALL = 2
  SNAKE = 3

LABEL2COLOR = {
  Label.FOOD: FOOD_COLOR,
  Label.WALL: WHITE,
  Label.SNAKE: SNAKE_COLOR
}

class SnakeDirection:
  RIGHT = 0
  DOWN = 1
  LEFT = 2
  UP = 3

class Snake:
  def __init__(self, init_coord):
    self.body_directions = []
    self.coords = [init_coord]
    self.direction = SnakeDirection.RIGHT

  def grow(self):
    new_coord = move_coord(self.coords[-1], self.direction)
    self.body_directions.append(self.direction)
    self.coords.append(new_coord)

    return new_coord

  def move(self):
    self.grow()
    rear_coord = self.coords[0]
    del self.coords[0]
    del self.body_directions[0]
    return rear_coord

  def mapping(self, map):
    for coord in self.coords:
      map[coord] = Label.SNAKE

def move_coord(coord, direction):
  x = coord[0]
  y = coord[1]

  if direction == SnakeDirection.RIGHT:
    x += 1
  elif direction == SnakeDirection.DOWN:
    y += 1
  elif direction == SnakeDirection.LEFT:
    x -= 1
  elif direction == SnakeDirection.UP:
    y -= 1

  return (x,y)