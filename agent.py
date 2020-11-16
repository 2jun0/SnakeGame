from collections import deque
import random
from re import L
from typing import Sequence
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from snake import NUM_ACTIONS, NUM_LABELS
import numpy as np

class DQNAgent:
  def __init__(self, fc_state_size: int, cnn_state_size: int, action_size: int):
    # 상태와 행동의 크기 정의
    self.fc_state_size = fc_state_size
    self.cnn_state_size = cnn_state_size
    self.action_size = action_size

    # DQN 하이퍼파라미터
    self.epsilon = 1.0
    self.epsilon_decay = 0.999
    self.epsilon_min = 0.01
    self.learning_rate = 0.001
    self.discount_factor = 0.99
    self.batch_size = 64
    self.train_start = 1000

    # 모델과 타겟 모델 생성
    self.model = self._create_model()
    self.target_model = self._create_model()
    self.model.summary()

    # 리플레이 메모리 2000
    self.replay_memory = deque(maxlen=10000)

    # 타겟 업데이트 설정
    self.target_update_counter = 0
    self.target_update_freq = 100
    self.update_target_model()

    # 러닝 모드
    self.learning = False

  def _create_model(self):
    cnn_input = Input(shape=(self.cnn_state_size, self.cnn_state_size, NUM_LABELS), batch_size=self.batch_size)
    cnn_model = Conv2D(8, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform')(cnn_input)
    cnn_model = Conv2D(8, (2,2), padding='same', activation='relu', kernel_initializer='he_uniform')(cnn_model)
    cnn_model = Flatten()(cnn_model)

    fc_input = Input(shape=(self.fc_state_size), batch_size=self.batch_size)
    fc_model = Dense(64, activation='relu', kernel_initializer='he_uniform')(fc_input)
    fc_model = Dense(64, activation='relu', kernel_initializer='he_uniform')(fc_model)

    concat_model = concatenate([fc_model, cnn_model])
    concat_model = Dense(128, activation='relu', kernel_initializer='he_uniform')(concat_model)
    concat_model = Dense(self.action_size, activation='sigmoid', kernel_initializer='he_uniform')(concat_model)

    adam = Adam(lr=self.learning_rate)
    concat_model.compile(optimizer=adam, loss='mse')
    return concat_model

  def update_replay_memory(self, current_state, action, reward, next_state, done):
    self.replay_memory.append((current_state, action, reward, next_state, done))

  def get_action(self, state):
    if np.random.rand() <= self.epsilon and self.learning:
      return random.randrange(self.action_size)
    else:
      q_value = self.model.predict(state)
      return np.argmax(q_value[0])

  def train(self):
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

    # 현재 Q 값을 구한 뒤, 다음 Q 값을 구함
    # sample은 (current_state, action, reward, next_state, done)
    mini_batch = random.sample(self.replay_memory, self.batch_size) # 샘플들을 batch_size 만큼 무작위로 뽑는다.

    states = np.zeros((self.batch_size, self.state_size))
    next_states = np.zeros((self.batch_size, self.state_size))
    actions, rewards, dones = [], [], []

    for i in range(self.batch_size):
      states[i] = mini_batch[i][0]
      actions.append(mini_batch[i][1])
      rewards.append(mini_batch[i][2])
      next_states[i] = mini_batch[i][3]
      dones.append(mini_batch[i][4])

    # 현재 상태에 대한 모델, 타깃 모델의 Q값
    q_vals = self.model.predict(states)
    next_q_vals = self.target_model.predict(next_states)

    # Q 값 업데이트
    for i, (state, action, reward, next_state, done) in enumerate(mini_batch):
      if done:
        q_vals[i, action] = reward
      else:
        q_vals[i, action] = reward + self.discount_factor * np.max(next_q_vals[i])
    
    # 모델 학습
    hist = self.model.fit(states, q_vals, batch_size=self.batch_size, epochs=1, verbose=0)
    loss = hist.history['loss'][0]
    return loss

  def update_target_model(self):
    """타깃 모델을 모델의 가중치로 업데이트"""
    self.target_model.set_weights(self.model.get_weights())

  def increase_target_update_counter(self):
    self.target_update_counter += 1
    if self.target_update_counter >= self.target_update_freq:
      self.target_model.set_weights(self.model.get_weights())
      self.target_update_counter = 0

  def save(self, model_filepath, target_model_filepath):
    self.model.save(model_filepath)
    self.target_model.save(target_model_filepath)

  def load(self, model_filepath, target_model_filepath):
    self.model = keras.models.load_model(model_filepath)
    self.target_model = keras.models.load_model(target_model_filepath)

    # adam = Adam(lr=self.learning_rate)
    # self.model.compile(optimizer=adam, loss='mse')

    # adam = Adam(lr=self.learning_rate)
    # self.target_model.compile(optimizer=adam, loss='mse')

  def save_plot_model(self, path):
    plot_model(self.model, path)