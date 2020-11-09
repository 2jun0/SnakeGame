import os
import random
import numpy as np
import pygame
import tensorflow as tf
from tqdm import tqdm
from config import SNAKE_VISION_MAP_RANGE
from snake import NUM_ACTIONS
from agent import DQNAgent

class DQNTrainer:
	def __init__(self,
				env,
			  num_episode=10000, 
			  initial_epsilon=1.,
			  min_epsilon=0.1,
			  exploration_ratio=0.5,
			  max_steps=100,
			  enable_draw =True,
			  draw_freq = 100,
			  draw_fps=20,
			  enable_save=True,
			  save_dir='checkpoints',
			  save_freq=500,
			  gamma=0.99,
			  batch_size=256,
			  min_replay_memory_size=256,
			  replay_memory_size=1000,
			  target_update_freq=1000,
			  seed=42):
		self.set_random_seed(seed)
	
		self.num_episode = num_episode
		self.max_steps = max_steps
		self.epsilon = initial_epsilon
		self.min_epsilon = min_epsilon
		self.exploration_ratio = exploration_ratio
		self.draw_freq = draw_freq
		self.enable_draw = enable_draw
		self.draw_fps = draw_fps
		self.enable_save = enable_save
		self.save_dir = save_dir
		self.save_freq = save_freq

		if enable_save and not os.path.exists(save_dir):
			os.makedirs(save_dir)

		self.agent = DQNAgent(
			cnn_input_size = SNAKE_VISION_MAP_RANGE,
			fc_input_size = 8,
			gamma = gamma,
			batch_size = batch_size,
			min_replay_memory_size = min_replay_memory_size,
			replay_memory_size = replay_memory_size,
			target_update_freq = target_update_freq
		)
		self.env = env
		self.current_episode = 0
		self.max_average_length = 0

		self.max_score = 0

		self.epsilon_decay = (initial_epsilon-min_epsilon)/(exploration_ratio*num_episode)

	def set_random_seed(self, seed):
		random.seed(seed)
		np.random.seed(seed)

	def train(self):
		pbar = tqdm(initial=self.current_episode, total=self.num_episode, unit='episodes')

		# num_episode만큼 반복
		while self.current_episode < self.num_episode:
			current_state = self.env.reset()

			done = False
			steps = 0
			# 게임이 끝나거나, 게임 턴이 만료되었을때까지
			while not done and steps < self.max_steps:
				if random.random() > self.epsilon:
					action = np.argmax(self.agent.get_q_values(np.array([current_state])))
				else:
					action = np.random.randint(NUM_ACTIONS)
					
				next_state, reward, done = self.env.step(action)

				if self.max_score < reward:
					self.max_score = reward
					print('\nnew score! ', self.max_score)
					self.preview(20)

				self.agent.update_replay_memory(current_state, action, reward, next_state, done)
				
				if reward != 0:
					self.agent.train()
				
				current_state = next_state
				steps += 1

			self.agent.increase_target_update_counter()

			self.epsilon = max(self.epsilon-self.epsilon_decay, self.min_epsilon)

			self.current_episode += 1

			# 모델 저장
			if self.enable_save and self.current_episode % self.save_freq == 0:
				self.save(str(self.current_episode))

			pbar.update(1)

			if self.enable_draw and self.current_episode % self.draw_freq == 0:
				self.preview(self.draw_fps)

	def preview(self, draw_fps):
		self.env.painter.open_display()
		current_state = self.env.reset()

		step = 0
		while not self.env.is_game_over and step < 500:
			for event in pygame.event.get():
				if event.type==pygame.QUIT:
					self.env.end_display()
					return

			action = np.argmax(self.agent.get_q_values(np.array([current_state])))

			current_state, reward, done = self.env.step(action)

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
				