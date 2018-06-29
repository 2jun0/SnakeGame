"""
# project name : snake game
# programmer : 이준영 (soo28819@naver.com)
# description : snake game using pygame in python
# 가독성을 위주로 작성하였음.
"""
import pygame, sys
import random

pygame.init()

"""
Const
"""
#window size Const
DISPLAY_WIDTH = 400
DISPLAY_HEIGHT = 320
ONE_BLOCK_SIZE = 10
GAME_MAP_WIDTH = 36
GAME_MAP_HEIGHT = 26
GAME_MAP_X = (DISPLAY_WIDTH - GAME_MAP_WIDTH * ONE_BLOCK_SIZE)/2
GAME_MAP_Y = (DISPLAY_HEIGHT - GAME_MAP_HEIGHT * ONE_BLOCK_SIZE)/2 + 10

#direction Const
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

#color Const
WHITE = (255,255,255)
FOOD_COLOR = (0xff, 0x0,0x66)
SNAKE_COLOR = (0x66,0x66,0xff)
SNAKE_HEAD_COLOR = (0x33,0x33,0xff)
BACKGROUND_COLOR = (0x11,0x11,0x11)
"""
"""

#set window size
DISPLAYSURF = pygame.display.set_mode((DISPLAY_WIDTH,DISPLAY_HEIGHT))
#set window menu bar name
pygame.display.set_caption('Snake game')

#snake
snake_coords = []
snake_move_direction = -1
#food
food_coord = None

#game play
score = 0
is_game_over = False

def main():
	initGame()

	while True:
		key = -1

		if is_game_over:
			for event in pygame.event.get():
				if event.type is pygame.QUIT:
					pygame.quit()
					sys.exit()

				if event.type is pygame.KEYDOWN:
					initGame()
					continue

			drawGameOver()
			pygame.display.update()

		else:
			for event in pygame.event.get():
				if event.type is pygame.QUIT:
					pygame.quit()
					sys.exit()

				key = getKey(event)
				if key != -1:
					print("key : "+str(key))
			moveSnake(key)
			drawGame()
			pygame.display.update()
	
		pygame.time.Clock().tick(30)

def initGame():
	global is_game_over, snake_move_direction, score

	score = 0
	is_game_over = False
	
	snake_move_direction = -1
	snake_coords.clear()
	snake_coords.append((GAME_MAP_WIDTH/2-2, GAME_MAP_HEIGHT/2))
	snake_coords.insert(0,(snake_coords[0][0]+1, snake_coords[0][1]))
	snake_coords.insert(0,(snake_coords[1][0]+1, snake_coords[1][1]))
	randomFoodAppear()

def drawGameOver():
	#set window background color
	DISPLAYSURF.fill(BACKGROUND_COLOR)
	#draw text
	fontObj = pygame.font.Font("C:\\Windows\\Fonts\\HoonWhitecatR.ttf", 30)
	mainText = fontObj.render("스네이크 게임", True, WHITE)
	DISPLAYSURF.blit(mainText, (DISPLAY_WIDTH/2 - 60, 0))

	gameOverText = fontObj.render("GAME OVER", True, WHITE)
	DISPLAYSURF.blit(gameOverText, (DISPLAY_WIDTH/2 - 70, GAME_MAP_Y + GAME_MAP_HEIGHT * ONE_BLOCK_SIZE /2 - 50))

	fontObj = pygame.font.Font("C:\\Windows\\Fonts\\HoonWhitecatR.ttf", 20)
	scoreText = fontObj.render("당신의 점수는 " + str(score), True, WHITE)
	DISPLAYSURF.blit(scoreText, (DISPLAY_WIDTH/2 - 50, GAME_MAP_Y + GAME_MAP_HEIGHT * ONE_BLOCK_SIZE /2))

	resetText = fontObj.render("다시시작하려면 아무키나 눌러", True, WHITE)
	DISPLAYSURF.blit(resetText, (DISPLAY_WIDTH/2 - 90, GAME_MAP_Y + GAME_MAP_HEIGHT * ONE_BLOCK_SIZE /2 + 40))

	pygame.draw.rect(DISPLAYSURF, WHITE, (GAME_MAP_X - 10 , GAME_MAP_Y - 10, GAME_MAP_WIDTH * ONE_BLOCK_SIZE + 20, GAME_MAP_HEIGHT * ONE_BLOCK_SIZE + 20),1)

def drawGame():
	#set window background color
	DISPLAYSURF.fill(BACKGROUND_COLOR)
	#draw text
	fontObj = pygame.font.Font("C:\\Windows\\Fonts\\HoonWhitecatR.ttf", 30)
	mainText = fontObj.render("스네이크 게임", True, WHITE)
	DISPLAYSURF.blit(mainText, (DISPLAY_WIDTH/2 - 60, 0))
	fontObj = pygame.font.Font("C:\\Windows\\Fonts\\HoonWhitecatR.ttf", 20)
	scoreText = fontObj.render("점수 : "+str(score), True, WHITE)
	DISPLAYSURF.blit(scoreText, (DISPLAY_WIDTH - 60, 5))

	pygame.draw.rect(DISPLAYSURF, WHITE, (GAME_MAP_X - 10 , GAME_MAP_Y - 10, GAME_MAP_WIDTH * ONE_BLOCK_SIZE + 20, GAME_MAP_HEIGHT * ONE_BLOCK_SIZE + 20),1)
	drawSnake()
	drawFood()

def drawSnake():
	for coord in snake_coords:
		drawRect(coord, SNAKE_COLOR)

def drawFood():
	drawRect(food_coord, FOOD_COLOR)

def checkCollision(coord):
	return coord in snake_coords or ((coord[0] < 0 or coord[1] < 0) or (coord[0] >= GAME_MAP_WIDTH or coord[1] >= GAME_MAP_HEIGHT))

def randomFoodAppear():
	global food_coord

	coord = (random.randrange(0,GAME_MAP_WIDTH), random.randrange(0,GAME_MAP_HEIGHT))
	while coord in snake_coords:
		x = coord[0] + 1
		y = coord[1] + x/GAME_MAP_WIDTH%GAME_MAP_HEIGHT
		x %= GAME_MAP_WIDTH

	food_coord = coord

def moveSnake(direction):
	global snake_move_direction, score, is_game_over
	new_head = None
	print(snake_move_direction)
	if direction is LEFT and (snake_move_direction is not RIGHT and snake_move_direction is not -1):
		new_head = (snake_coords[0][0] - 1, snake_coords[0][1])
		snake_move_direction = direction
	elif direction is RIGHT and snake_move_direction is not LEFT:
		new_head = (snake_coords[0][0] + 1, snake_coords[0][1])
		snake_move_direction = direction
	elif direction is UP and snake_move_direction is not DOWN:
		new_head = (snake_coords[0][0], snake_coords[0][1] - 1)
		snake_move_direction = direction
	elif direction is DOWN and snake_move_direction is not UP:
		new_head = (snake_coords[0][0], snake_coords[0][1] + 1)
		snake_move_direction = direction
	else:
		if snake_move_direction is LEFT:
			new_head = (snake_coords[0][0] - 1, snake_coords[0][1])
		elif snake_move_direction is RIGHT:
			new_head = (snake_coords[0][0] + 1, snake_coords[0][1])
		elif snake_move_direction is UP:
			new_head = (snake_coords[0][0], snake_coords[0][1] - 1)
		elif snake_move_direction is DOWN:
			new_head = (snake_coords[0][0], snake_coords[0][1] + 1)
		else:
			#아무것도 안함
			return

		if new_head[0] == food_coord[0] and new_head[1] == food_coord[1]:
			# eat food and don't delete tail
			score += 1
			randomFoodAppear()
		else:
			if checkCollision(new_head):
				# game over!
				print("new head coord : " + str(new_head))
				for coord in snake_coords:
					print("snake coord : " + str(coord))
				print("direction : " + str(snake_move_direction))
				is_game_over = True
				print("state : game over")
				return

			del snake_coords[-1]

		snake_coords.insert(0, new_head)

def getKey(event):
	if event.type == pygame.KEYDOWN:
		if event.key == pygame.K_UP:
			return UP
		elif event.key == pygame.K_DOWN:
			return DOWN
		elif event.key == pygame.K_LEFT:
			return LEFT
		elif event.key == pygame.K_RIGHT:
			return RIGHT
	return -1

def drawRect(coord, color):
	temp = gameCoordToDisplayCoord(coord)
	pygame.draw.rect(DISPLAYSURF, color, (temp[0], temp[1], ONE_BLOCK_SIZE, ONE_BLOCK_SIZE))

def gameCoordToDisplayCoord(coord):
	return (coord[0] * ONE_BLOCK_SIZE + GAME_MAP_X, coord[1] * ONE_BLOCK_SIZE + GAME_MAP_Y)

if __name__=="__main__":
	main()
