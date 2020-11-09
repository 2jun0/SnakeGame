import pygame, sys
from config import *
from snake import Label, LABEL2COLOR

class Painter:
	def open_display(self):
		pygame.init()
		#set window size
		self.DISPLAYSURF = pygame.display.set_mode((DISPLAY_WIDTH,DISPLAY_HEIGHT))
		#set window menu bar name
		pygame.display.set_caption('Snake game')

	def close_display(self):
		pygame.quit()
	
	def draw_game(self, score, map, snake_vision, food_indicator):
		#set window background color
		self.DISPLAYSURF.fill(BACKGROUND_COLOR)

		#fonts
		font20Obj = pygame.font.Font(FONT_PATH, 20)
		font25Obj = pygame.font.Font(FONT_PATH, 25)

		self._draw_map_frame()
	
		#draw text
		mainText = font25Obj.render("스네이크 게임", True, WHITE)
		self.DISPLAYSURF.blit(mainText, (DISPLAY_WIDTH/2 - 60, 0))
		
		scoreText = font20Obj.render("점수 : "+str(score), True, WHITE)
		self.DISPLAYSURF.blit(scoreText, (DISPLAY_WIDTH - 60, 0))
		
		# draw game objects
		for x in range(GAME_MAP_WIDTH):
			for y in range(GAME_MAP_HEIGHT):
				if map[x,y] != Label.BLANK:
					self._draw_block(coord=(x,y), color=LABEL2COLOR[map[x,y]], block_size=BLOCK_SIZE, srt_idx=(GAME_MAP_X, GAME_MAP_Y))

		self._draw_snake_vision(snake_vision)
		self._draw_food_indicator(food_indicator)

		pygame.display.update()

	def draw_game_over(self, score):
		#set window background color
		self.DISPLAYSURF.fill(BACKGROUND_COLOR)
		#fonts
		font20Obj = pygame.font.Font(FONT_PATH, 20)
		font25Obj = pygame.font.Font(FONT_PATH, 25)

		self._draw_map_frame()
		self._draw_snake_vision_frame()
		self._draw_food_indicator_frame()

		#draw text
		mainText = font25Obj.render("스네이크 게임", True, WHITE)
		self.DISPLAYSURF.blit(mainText, (DISPLAY_WIDTH/2 - 60, 0))

		gameOverText = font25Obj.render("GAME OVER", True, WHITE)
		self.DISPLAYSURF.blit(gameOverText, (GAME_MAP_X+GAME_MAP_WIDTH*BLOCK_SIZE/2-60, GAME_MAP_Y + GAME_MAP_HEIGHT * BLOCK_SIZE /2 - 50))
		
		scoreText = font20Obj.render("당신의 점수는 " + str(score), True, WHITE)
		self.DISPLAYSURF.blit(scoreText, (GAME_MAP_X+GAME_MAP_WIDTH*BLOCK_SIZE/2-60, GAME_MAP_Y + GAME_MAP_HEIGHT * BLOCK_SIZE /2))

		resetText = font20Obj.render("다시시작하려면 아무키나 눌러", True, WHITE)
		self.DISPLAYSURF.blit(resetText, (GAME_MAP_X+GAME_MAP_WIDTH*BLOCK_SIZE/2-100, GAME_MAP_Y + GAME_MAP_HEIGHT * BLOCK_SIZE /2 + 40))

		pygame.display.update()

	def _draw_map_frame(self):
		#draw map frame
		pygame.draw.rect(self.DISPLAYSURF, WHITE, (GAME_MAP_X, GAME_MAP_Y, GAME_MAP_WIDTH * BLOCK_SIZE, GAME_MAP_HEIGHT * BLOCK_SIZE),1)
	def _draw_snake_vision_frame(self):
		# draw snake vision frame
		pygame.draw.rect(self.DISPLAYSURF, WHITE, (
			SNAKE_VISION_MAP_X, SNAKE_VISION_MAP_Y,
			SNAKE_VISION_MAP_SIZE[0], SNAKE_VISION_MAP_SIZE[1]), 1)
	def _draw_food_indicator_frame(self):
		# draw food indicator frame
		pygame.draw.rect(self.DISPLAYSURF, WHITE, (
			FOOD_INDICATOR_X, FOOD_INDICATOR_Y,
			FOOD_INDICATOR_SIZE[0], FOOD_INDICATOR_SIZE[1]), 1)

	def _draw_snake_vision(self, snake_vision):
		self._draw_snake_vision_frame()

		# draw objects in snake vision
		for x in range(SNAKE_VISION_MAP_RANGE[0]):
			for y in range(SNAKE_VISION_MAP_RANGE[1]):
				if snake_vision[x,y] != Label.BLANK:
					self._draw_block(coord=(x,y), color=LABEL2COLOR[snake_vision[x,y]], block_size=SNAKE_VISION_BLOCK_SIZE, srt_idx=(SNAKE_VISION_MAP_X, SNAKE_VISION_MAP_Y))

	def _draw_food_indicator(self, food_indicator):
		self._draw_food_indicator_frame()

		# draw food indicator
		angle_coords = [(1,0),(0,0),(0,1),(0,2),(1,2),(2,2),(2,1),(2,0)]
		for angle_coord, food_angle_idx in zip(angle_coords, food_indicator):
			if food_angle_idx:
				self._draw_block(coord=angle_coord, color=LABEL2COLOR[Label.FOOD], block_size=FOOD_INDICATOR_BLOCK_SIZE, srt_idx=(FOOD_INDICATOR_X, FOOD_INDICATOR_Y))
			else:
				self._draw_block(coord=angle_coord, color=WHITE, block_size=FOOD_INDICATOR_BLOCK_SIZE, srt_idx=(FOOD_INDICATOR_X, FOOD_INDICATOR_Y))

	def _draw_block(self, coord, color, block_size, srt_idx=(0,0)):
		temp = self._game_coord2display_coord(coord, block_size, srt_idx)
		pygame.draw.rect(self.DISPLAYSURF, color, (temp[0], temp[1], block_size, block_size))

	def _game_coord2display_coord(self, coord, block_size, srt_idx=(0,0)):
		return (coord[0] * block_size + srt_idx[0], coord[1] * block_size + srt_idx[1])
