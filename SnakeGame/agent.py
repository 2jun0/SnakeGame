from collections import deque
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, Input, concatenate
from tensorflow.keras.optimizers import Adam
from snake import NUM_ACTIONS, NUM_LABELS
import numpy as np

# import wandb
# from wandb.keras import WandbCallback
# wandb.init(config={"hyper": "parameter"})

class DQNAgent:
	def __init__(self, cnn_input_size, fc_input_size, batch_size, gamma, min_replay_memory_size, replay_memory_size, target_update_freq):
		self.cnn_input_size = cnn_input_size
		self.fc_input_size = fc_input_size
		self.batch_size = batch_size
		self.gamma = gamma
		self.min_replay_memory_size = min_replay_memory_size
		self.target_update_freq = target_update_freq

		self.model = self._create_model()
		self.target_model = self._create_model()
		self.target_model.set_weights(self.model.get_weights())
		self.model.summary()

		self.replay_memory = deque(maxlen=replay_memory_size)
		self.target_update_counter = 0

	def _create_model(self):
		fc_input = Input(shape=(self.fc_input_size))
		fc_model = Dense(128, activation='relu')(fc_input)

		fc_model = Dense(128, activation='relu')(fc_model)
		fc_model = Dropout(0.1)(fc_model)
		fc_model = Dense(NUM_ACTIONS)(fc_model)

		adam = Adam(lr=0.1)
		model = Model(inputs=[fc_input], outputs=fc_model)
		model.compile(optimizer=adam, loss='mse')
		return model

	def update_replay_memory(self, current_state, action, reward, next_state, done):
		self.replay_memory.append((current_state, action, reward, next_state, done))

	def get_q_values(self, x):
		return self.model.predict(x)

	def train(self):
		# 최소 replay memory를 만족해야함.
		if len(self.replay_memory) < self.min_replay_memory_size:
			return

		# 현재 Q 값을 구한 뒤, 다음 Q 값을 구함
		# sample은 (current_state, action, reward, next_state, done)
		samples = random.sample(self.replay_memory, self.batch_size) # 샘플들을 batch_size 만큼 무작위로 뽑는다.
		current_input = np.stack(sample[0] for sample in samples) # 샘플들을 stack 자료구조로 변환
		current_q_values = self.model.predict(current_input)
		next_input = np.stack(sample[3] for sample in samples)
		next_q_values = self.target_model.predict(next_input)

		# Q 값 업데이트
		for i, (current_state, action, reward, _, done) in enumerate(samples):
			if done:
				next_q_value = reward
			else:
				next_q_value = reward + self.gamma * np.max(next_q_values[i])
			current_q_values[i, action] = next_q_value

		# 모델 학습
		hist = self.model.fit(current_input, current_q_values, batch_size=self.batch_size, verbose=0, shuffle=False)#, callbacks=[WandbCallback()])
		loss = hist.history['loss'][0]
		return loss

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