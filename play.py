import argparse
import pygame
import sys
from SnakeGame import SnakeGame
from SnakeGame import getKey

parser = argparse.ArgumentParser()

args = parser.parse_args()

def main():
  freeze = True
  game = SnakeGame()
  game.reset()
  
  while True:
    key = -1

    if game.is_game_over:
      for event in pygame.event.get():
        if event.type is pygame.QUIT:
          pygame.quit()
          sys.exit()

        if event.type is pygame.KEYDOWN:
          game.reset()
          freeze = True
          continue
    else:
      for event in pygame.event.get():
        if event.type is pygame.QUIT:
          pygame.quit()
          sys.exit()

        key = getKey(event)
          
      if key != -1:
        game.turn_snake_to(key)
        freeze = False

      if not freeze:
        game.move_snake_forward()

    game.draw()
    pygame.time.Clock().tick(20)

if __name__=="__main__":
  main()