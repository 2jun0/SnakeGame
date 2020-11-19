import os
import random
import numpy as np
import pygame
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from snake import NUM_ACTIONS, NUM_LABELS
from agent import DQNAgent

CNN_STATE_SHAPE = (5, 5, NUM_LABELS)
FC_STATE_SHAPE = (8,)

class DQNTrainer:
  def __init__(self,
        env,
        name:str,
        num_episode=300, 
        max_steps=100,
        enable_draw =True,
        draw_freq = 100,
        draw_fps=20,
        enable_save=True,
        save_freq=100,
        seed=42):
    self.set_random_seed(seed)
  
    self.name = name
    self.num_episode = num_episode
    self.max_steps = max_steps
    self.draw_freq = draw_freq
    self.enable_draw = enable_draw
    self.draw_fps = draw_fps
    self.enable_save = enable_save
    self.save_freq = save_freq

    self.agent = DQNAgent(
      cnn_state_shape=CNN_STATE_SHAPE,
      fc_state_shape=FC_STATE_SHAPE,
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
      # state를 cnn과 fc로 분리
      cnn_states, fc_states = state
      cnn_states = np.reshape(cnn_states, (1,) + CNN_STATE_SHAPE)
      fc_states = np.reshape(fc_states, (1,) + FC_STATE_SHAPE)

      done = False
      steps = 0
      score = 0

      # 게임이 끝나거나, 게임 턴이 만료되었을때까지
      while not done and steps < self.max_steps:
        action = self.agent.get_action([cnn_states, fc_states])
        next_state, reward, done = self.env.step(action)
        # next_state를 cnn과 fc로 분리
        next_cnn_states, next_fc_states = next_state
        next_cnn_states = np.reshape(next_cnn_states, (1,) + CNN_STATE_SHAPE)
        next_fc_states = np.reshape(next_fc_states, (1,) + FC_STATE_SHAPE)
        
        # 에피소드가 중간에 끝나면 -10보상
        reward = reward if not done else -10

        # 리플레이 메모리에 결과 저장
        self.agent.update_replay_memory([cnn_states, fc_states], action, reward, [next_cnn_states, next_fc_states], done)

        if self.max_score < reward:
          self.max_score = reward
          print('\nnew score! ', self.max_score)
        
        # 매 타임스텝마다 학습
        if len(self.agent.replay_memory) >= self.agent.train_start:
          self.agent.train()

        if reward > 0:
          score += 1
        
        cnn_states = next_cnn_states
        fc_states = next_fc_states

        for event in pygame.event.get():
          # 종료
          if event.type is pygame.QUIT:
            self.save('tmp')
            self.summary(episodes=range(start_episode, current_episode), scores=scores)
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

      scores.append(score if score > 0 else 0)

      # 모델 저장
      if self.enable_save and current_episode % self.save_freq == 0:
        self.save(str(current_episode))
        self.summary(episodes=range(start_episode, current_episode), scores=scores)

      pbar.update(1)

      # if self.enable_draw and current_episode % self.draw_freq == 0:
      #   self.preview(self.draw_fps)

  def preview(self, draw_fps: int, looping: bool = False):
    self.env.painter.open_display()

    self.agent.learning = False

    while True:
      state = self.env.reset()
      state = np.reshape(state, [1, STATE_SIZE])

      done = False

      while not done:
        for event in pygame.event.get():
          if event.type==pygame.QUIT:
            pygame.quit()
            sys.exit()

        action = self.agent.get_action(state)
        next_state, reward, done = self.env.step(action)
        next_state = np.reshape(next_state, [1, STATE_SIZE])
        state = next_state

        self.env.draw()
    
        pygame.time.Clock().tick(draw_fps)

      if not looping:
        break

    self.env.painter.close_display()

  def save(self, suffix):
    self.agent.save(
        f'checkpoints/model_{self.name}_{suffix}.h5',
        f'checkpoints/target_model_{self.name}_{suffix}.h5'
    )
    

  def load(self, suffix):
    self.agent.load(
        f'checkpoints/model_{self.name}_{suffix}.h5',
        f'checkpoints/target_model_{self.name}_{suffix}.h5'
    )

  def summary(self, episodes, scores):
    self.agent.save_plot_model(f'plots/model_{self.name}.png')

    plt.plot(episodes, scores, 'b')
    plt.savefig(f'plots/scores_{self.name}_{episodes[0]}_{episodes[-1]}.png')