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
DIRECTIONS = {
	'left': 0,
	'right': 1,
	'up': 2,
	'down': 3}

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

#others
score = 0
is_game_over = False
FONT_PATH = 'NanumBarunpenR.ttf'

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
				turnSnake(key)
				print("key : "+str(key))

			moveSnake()
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
	snake_coords.insert(0,(snake_coords[0][0], snake_coords[0][1]))
	snake_coords.insert(0,(snake_coords[1][0], snake_coords[1][1]))
	randomFoodAppear()

def drawGameOver():
	#set window background color
	DISPLAYSURF.fill(BACKGROUND_COLOR)
	#draw text
	font25Obj = pygame.font.Font(FONT_PATH, 25)
	mainText = font25Obj.render("스네이크 게임", True, WHITE)
	DISPLAYSURF.blit(mainText, (DISPLAY_WIDTH/2 - 60, 0))

	gameOverText = font25Obj.render("GAME OVER", True, WHITE)
	DISPLAYSURF.blit(gameOverText, (DISPLAY_WIDTH/2 - 60, GAME_MAP_Y + GAME_MAP_HEIGHT * ONE_BLOCK_SIZE /2 - 50))

	font20Obj = pygame.font.Font(FONT_PATH, 20)
	scoreText = font20Obj.render("당신의 점수는 " + str(score), True, WHITE)
	DISPLAYSURF.blit(scoreText, (DISPLAY_WIDTH/2 - 60, GAME_MAP_Y + GAME_MAP_HEIGHT * ONE_BLOCK_SIZE /2))

	resetText = font20Obj.render("다시시작하려면 아무키나 눌러", True, WHITE)
	DISPLAYSURF.blit(resetText, (DISPLAY_WIDTH/2 - 100, GAME_MAP_Y + GAME_MAP_HEIGHT * ONE_BLOCK_SIZE /2 + 40))

	pygame.draw.rect(DISPLAYSURF, WHITE, (GAME_MAP_X - 10 , GAME_MAP_Y - 10, GAME_MAP_WIDTH * ONE_BLOCK_SIZE + 20, GAME_MAP_HEIGHT * ONE_BLOCK_SIZE + 20),1)

def drawGame():
	#set window background color
	DISPLAYSURF.fill(BACKGROUND_COLOR)
	#draw text
	font25Obj = pygame.font.Font(FONT_PATH, 25)
	mainText = font25Obj.render("스네이크 게임", True, WHITE)
	DISPLAYSURF.blit(mainText, (DISPLAY_WIDTH/2 - 60, 0))
	font20Obj = pygame.font.Font(FONT_PATH, 20)
	scoreText = font20Obj.render("점수 : "+str(score), True, WHITE)
	DISPLAYSURF.blit(scoreText, (DISPLAY_WIDTH - 60, 0))

	pygame.draw.rect(DISPLAYSURF, WHITE, (GAME_MAP_X - 10 , GAME_MAP_Y - 10, GAME_MAP_WIDTH * ONE_BLOCK_SIZE + 20, GAME_MAP_HEIGHT * ONE_BLOCK_SIZE + 20),1)
	drawSnake()
	drawFood()

def drawSnake():
	# 뱀을 그리는 함수
	for coord in snake_coords:
		drawRect(coord, SNAKE_COLOR)

def drawFood():
	# 음식을 그리는 함수
	drawRect(food_coord, FOOD_COLOR)

def checkCollision(coord):
	# 충돌체크 함수
	return coord in snake_coords or ((coord[0] < 0 or coord[1] < 0) or (coord[0] >= GAME_MAP_WIDTH or coord[1] >= GAME_MAP_HEIGHT))

def randomFoodAppear():
	# 음식을 랜덤한 위치에 발생시키는 함수
	global food_coord

	coord = (random.randrange(0,GAME_MAP_WIDTH), random.randrange(0,GAME_MAP_HEIGHT))
	while coord in snake_coords:
		x = coord[0] + 1
		y = coord[1] + x/GAME_MAP_WIDTH%GAME_MAP_HEIGHT
		x %= GAME_MAP_WIDTH

	food_coord = coord

def turnSnake(direction):
	# 뱀의 이동방향을 변경하는 함수
	# direction은 DIRECTIONS['left', 'right', 'up', 'down']
	global snake_move_direction
	assert direction in DIRECTIONS.values()

	 # 뱀이 정반대 방향으로 이동방향을 변경하려는 경우 -> 함수 종료
	if direction is DIRECTIONS['left'] and snake_move_direction is DIRECTIONS['right']:
		return
	if direction is DIRECTIONS['right'] and snake_move_direction is DIRECTIONS['left']:
		return
	if direction is DIRECTIONS['up'] and snake_move_direction is DIRECTIONS['down']:
		return
	if direction is DIRECTIONS['down'] and snake_move_direction is DIRECTIONS['up']:
		return

	print('The snake turns to : ', direction)
	snake_move_direction = direction

def moveSnake():
	# 뱀을 움직이는 함수.
	global score, is_game_over
	new_head = None

	if snake_move_direction is DIRECTIONS['left']:
		new_head = (snake_coords[0][0] - 1, snake_coords[0][1])
	elif snake_move_direction is DIRECTIONS['right']:
		new_head = (snake_coords[0][0] + 1, snake_coords[0][1])
	elif snake_move_direction is DIRECTIONS['up']:
		new_head = (snake_coords[0][0], snake_coords[0][1] - 1)
	elif snake_move_direction is DIRECTIONS['down']:
		new_head = (snake_coords[0][0], snake_coords[0][1] + 1)
	else:
		# 정지
		return 

	if new_head == food_coord:
		# eat food and don't delete tail
		score += 1
		randomFoodAppear()
	else:
		# don't eat food and delete tail
		if checkCollision(new_head):
			# game over!
			is_game_over = True
			print('-Game Over-')
			return

		del snake_coords[-1]

	print('The snake moves to  : (%d,%d)' % (new_head[0], new_head[1]))
	snake_coords.insert(0, new_head)

		

def getKey(event):
	if event.type == pygame.KEYDOWN:
		if event.key == pygame.K_UP:
			return DIRECTIONS['up']
		elif event.key == pygame.K_DOWN:
			return DIRECTIONS['down']
		elif event.key == pygame.K_LEFT:
			return DIRECTIONS['left']
		elif event.key == pygame.K_RIGHT:
			return DIRECTIONS['right']
	return -1

def drawRect(coord, color):
	temp = gameCoordToDisplayCoord(coord)
	pygame.draw.rect(DISPLAYSURF, color, (temp[0], temp[1], ONE_BLOCK_SIZE, ONE_BLOCK_SIZE))

def gameCoordToDisplayCoord(coord):
	return (coord[0] * ONE_BLOCK_SIZE + GAME_MAP_X, coord[1] * ONE_BLOCK_SIZE + GAME_MAP_Y)

if __name__=="__main__":
	main()
