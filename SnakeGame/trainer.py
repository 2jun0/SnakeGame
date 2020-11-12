import os
import random
import numpy as np
import pygame
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from snake import NUM_ACTIONS
from agent import DQNAgent

STATE_SIZE = 8 + 4

class DQNTrainer:
	def __init__(self,
				env,
			  num_episode=10000, 
			  max_steps=500,
			  enable_draw =True,
			  draw_freq = 100,
			  draw_fps=20,
			  enable_save=True,
			  save_dir='checkpoints',
			  save_freq=500,
			  seed=42):
		self.set_random_seed(seed)
	
		self.num_episode = num_episode
		self.max_steps = max_steps
		self.draw_freq = draw_freq
		self.enable_draw = enable_draw
		self.draw_fps = draw_fps
		self.enable_save = enable_save
		self.save_dir = save_dir
		self.save_freq = save_freq

		if enable_save and not os.path.exists(save_dir):
			os.makedirs(save_dir)

		self.agent = DQNAgent(
			state_size=STATE_SIZE,
			action_size=NUM_ACTIONS
		)
		self.env = env
		self.max_average_length = 0

		self.max_score = 0

	def set_random_seed(self, seed):
		random.seed(seed)
		np.random.seed(seed)

	def train(self, start_episode=0):
		self.env.painter.open_display(training=True)
		self.agent.learning = True

		current_episode = start_episode
		scores = []

		pbar = tqdm(initial=current_episode, total=self.num_episode, unit='episodes')

		# num_episode만큼 반복
		while current_episode < self.num_episode:
			state = self.env.reset()
			state = np.reshape(state, [1, STATE_SIZE])

			done = False
			steps = 0
			score = 0

			# 게임이 끝나거나, 게임 턴이 만료되었을때까지
			while not done and steps < self.max_steps:
				action = self.agent.get_action(state)
				next_state, reward, done = self.env.step(action)
				next_state = np.reshape(next_state, [1, STATE_SIZE])
				# 에피소드가 중간에 끝나면 -100보상
				reward = reward if not done else -100

				# 리플레이 메모리에 결과 저장
				self.agent.update_replay_memory(state, action, reward, next_state, done)

				if self.max_score < reward:
					self.max_score = reward
					print('\nnew score! ', self.max_score)
				
				# 매 타임스텝마다 학습
				if len(self.agent.replay_memory) >= self.agent.train_start:
					self.agent.train()

				score += reward
				state = next_state

				for event in pygame.event.get():
					# 종료
					if event.type is pygame.QUIT:
						self.save('tmp')
						self.summary(episodes=range(start_episode, current_episode+1), scores=scores)
						pygame.quit()
						sys.exit()
					elif event.type == pygame.KEYDOWN:
						# 패스
						if event.key == pygame.K_p:
							done = True

				pygame.time.Clock().tick(240)

				self.env.painter.steps = steps
				self.env.draw()
				steps += 1

			self.agent.increase_target_update_counter()
			current_episode += 1
			scores.append(score)

			# 모델 저장
			if self.enable_save and current_episode % self.save_freq == 0:
				self.save(str(current_episode))

			pbar.update(1)

			# if self.enable_draw and current_episode % self.draw_freq == 0:
			# 	self.preview(self.draw_fps)

	def preview(self, draw_fps):
		self.env.painter.open_display()
		state = self.env.reset()
		state = np.reshape(state, [1, STATE_SIZE])

		step = 0
		self.agent.learning = False

		while not self.env.is_game_over and step < 500:
			for event in pygame.event.get():
				if event.type==pygame.QUIT:
					self.env.end_display()
					return

			action = self.agent.get_action(state)

			_, reward, done = self.env.step(action)

			self.env.draw()
	
			pygame.time.Clock().tick(draw_fps)
			step += 1

		self.env.painter.close_display()

	def save(self, suffix):
		self.agent.save(
				self.save_dir+'/model_{}.h5'.format(suffix),
				self.save_dir+'/target_model_{}.h5'.format(suffix)
		)

	def load(self, suffix):
		self.agent.load(
				self.save_dir+'/model_{}.h5'.format(suffix),
				self.save_dir+'/target_model_{}.h5'.format(suffix)
		)

	def summary(self, episodes, scores):
		plt.plot(episodes, scores, 'b')
		plt.savefig('save_graph/pot.png')
