"""
# project name : snake game
# programmer : 이준영 (soo28819@naver.com)
# description : snake game using pygame in python
# 가독성을 위주로 작성하였음.
"""
import pygame, sys
import random
import numpy as np
from config import *
from snake import Snake, Label, SnakeDirection, move_coord, NUM_LABELS
from painter import Painter
import math

class SnakeGame:
  def __init__(self, init_display=True):
    self.painter = Painter()

    if init_display:
      self.painter.open_display()

    self.reset()

  def reset(self):
    assert INITIAL_SNAKE_SIZE >= 1
    # map
    self.map = np.full((GAME_MAP_WIDTH, GAME_MAP_HEIGHT), Label.BLANK, dtype=np.uint8)

    # snake
    self.snake = Snake((int(GAME_MAP_WIDTH/2-INITIAL_SNAKE_SIZE/2), int(GAME_MAP_HEIGHT/2)))
    for _ in range(INITIAL_SNAKE_SIZE-1):
      self.snake.grow()
    # mapping
    self.snake.mapping(self.map)

    # food
    self._random_food_appear()

    # others
    self.score = 0
    self.is_game_over = False
    
    return self.get_state()

  def draw(self):
    if self.is_game_over:
      self.painter.draw_game_over(score=self.score)
    else:
      self.painter.draw_game(score=self.score, map=self.map, snake_vision=self.get_snake_vision_map(), food_indicator=self.get_food_indicator_map())

  def step(self, direction):
    self.turn_snake_to(direction)

    reward, done = self.move_snake_forward()
    return self.get_state(), reward, done

  def get_snake_vision_map(self):
    srt_idx = (
      max(0, self.snake.coords[-1][0] - int(SNAKE_VISION_MAP_RANGE[0]/2)),
      max(0 ,self.snake.coords[-1][1] - int(SNAKE_VISION_MAP_RANGE[1]/2)))
    dest_idx = ( # dest는 포함하는 index가 아님.
      min(GAME_MAP_WIDTH, self.snake.coords[-1][0] + int(SNAKE_VISION_MAP_RANGE[0]/2) + 1),
      min(GAME_MAP_HEIGHT, self.snake.coords[-1][1] + int(SNAKE_VISION_MAP_RANGE[1]/2) + 1)) 

    snake_vision_map = np.full((SNAKE_VISION_MAP_RANGE[0], SNAKE_VISION_MAP_RANGE[1]), Label.WALL, dtype=np.uint8)

    for v_map_x, map_x in enumerate(range(srt_idx[0], dest_idx[0])):
      for v_map_y, map_y in enumerate(range(srt_idx[1], dest_idx[1])):
        snake_vision_map[v_map_x,v_map_y] = self.map[map_x, map_y]

    return snake_vision_map

  def get_food_indicator_map(self):
    angle = math.degrees(math.atan2(self.snake.coords[-1][0]-self.food_coord[0], self.snake.coords[-1][1]-self.food_coord[1]))
    angle += 360
    angle %= 360

    angles = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    closest_angle = 0
    closest_v = 100
    for ang in angles:
      temp = abs(angle - ang)
      if closest_v > temp:
        closest_v = temp
        closest_angle = ang

    closest_angle %= 360
    food_indicator_map = np.zeros((8), dtype=np.uint8)
    food_indicator_map[int(closest_angle/45)] = 1

    return food_indicator_map

  def get_state(self):
    food_indicator_map = self.get_food_indicator_map()
    snake_direction_map = np.zeros(4)
    snake_direction_map[self.snake.direction] = 1
    
    return np.concatenate((food_indicator_map, snake_direction_map))

  def move_snake_forward(self):
    # 뱀을 앞으로 움직이는 함수.
    new_head = move_coord(self.snake.coords[-1], self.snake.direction)

    if self._check_collision(new_head):
      # game over!
      self.is_game_over = True
      return -1, True
    
    if new_head == self.food_coord:
      # eat food and don't delete tail
      self.score += 1
      self._random_food_appear()
      self.snake.grow()
      self.snake.mapping(self.map)

      return self.score, False
    else:
      # don't eat food and delete tail
      self.map[self.snake.move()] = Label.BLANK
      self.snake.mapping(self.map)
      return 0, False

  def turn_snake_to(self, direction):
    # 뱀의 머리를 direciton으로 돌리는 함수
    if (self.snake.direction + direction)%2 != 0:
      self.snake.direction = direction
  
  def _check_collision(self, coord):
    # 충돌체크 함수
    return coord in self.snake.coords or ((coord[0] < 0 or coord[1] < 0) or (coord[0] >= GAME_MAP_WIDTH or coord[1] >= GAME_MAP_HEIGHT))

  def _random_food_appear(self):
    # 음식을 랜덤한 위치에 발생시키는 함수
    
    while True:
      x = random.randint(0,GAME_MAP_WIDTH-1)
      y = random.randint(0,GAME_MAP_HEIGHT-1)
      if self.map[x,y] == Label.BLANK:
        break

    self.food_coord = (x, y)
    self.map[self.food_coord] = Label.FOOD

def getKey(event):
  if event.type == pygame.KEYDOWN:
    if event.key == pygame.K_UP:
      return SnakeDirection.UP
    elif event.key == pygame.K_DOWN:
      return SnakeDirection.DOWN
    elif event.key == pygame.K_LEFT:
      return SnakeDirection.LEFT
    elif event.key == pygame.K_RIGHT:
      return SnakeDirection.RIGHT
  return -1