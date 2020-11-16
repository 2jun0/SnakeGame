import argparse
from SnakeGame import SnakeGame
from trainer import DQNTrainer

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, default=None)
parser.add_argument('-n', '--name', type=str, default='')

def main():
  args = parser.parse_args()

  game = SnakeGame(init_display=False)
  game.reset()
  trainer = DQNTrainer(env = game, name = args.name)

  if args.checkpoint:
    trainer.load(suffix=args.checkpoint)
    trainer.preview(10)
    trainer.train(int(args.checkpoint))
  else:
    trainer.train()

if __name__=="__main__":
  main()