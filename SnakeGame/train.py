import argparse
from SnakeGame import SnakeGame
from trainer import DQNTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None)

args = parser.parse_args()

def main():
	game = SnakeGame(init_display=False)
	game.reset()
	trainer = DQNTrainer(env = game)

	if args.checkpoint:
		trainer.load(suffix=args.checkpoint)
		trainer.preview(10)

	trainer.train()
	pass

if __name__=="__main__":
	main()