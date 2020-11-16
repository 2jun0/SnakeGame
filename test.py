import argparse
from SnakeGame import SnakeGame
from trainer import DQNTrainer

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, required=True)
parser.add_argument('-n', '--name', type=str, default='')

def main():
  args = parser.parse_args()

  game = SnakeGame(init_display=False)
  game.reset()
  trainer = DQNTrainer(env = game, name = args.name)
  trainer.load(suffix=args.checkpoint)
  trainer.preview(30, looping=True)

if __name__=="__main__":
  main()