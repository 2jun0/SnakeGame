#window size Const
# display
DISPLAY_WIDTH = 700
DISPLAY_HEIGHT = 700

# map
GAME_MAP_WIDTH = 50
GAME_MAP_HEIGHT = 50

BLOCK_SIZE = 5

GAME_MAP_X = int((DISPLAY_WIDTH - GAME_MAP_WIDTH * BLOCK_SIZE)/2) - 40
GAME_MAP_Y = int((DISPLAY_HEIGHT - GAME_MAP_HEIGHT * BLOCK_SIZE)/2 + 10)

SNAKE_VISION_MAP_RANGE = (5,5)
SNAKE_VISION_MAP_X = GAME_MAP_X + (GAME_MAP_WIDTH*BLOCK_SIZE) + 20
SNAKE_VISION_MAP_Y = GAME_MAP_Y
SNAKE_VISION_BLOCK_SIZE = 10
SNAKE_VISION_MAP_SIZE = (SNAKE_VISION_MAP_RANGE[0]*SNAKE_VISION_BLOCK_SIZE, SNAKE_VISION_MAP_RANGE[1]*SNAKE_VISION_BLOCK_SIZE)

FOOD_INDICATOR_X = SNAKE_VISION_MAP_X
FOOD_INDICATOR_Y = SNAKE_VISION_MAP_Y + SNAKE_VISION_MAP_SIZE[1] + 20
FOOD_INDICATOR_BLOCK_SIZE = int(SNAKE_VISION_MAP_SIZE[0]/3)
FOOD_INDICATOR_SIZE = (3*FOOD_INDICATOR_BLOCK_SIZE, 3*FOOD_INDICATOR_BLOCK_SIZE)

INITIAL_SNAKE_SIZE = 2
FONT_PATH = 'NanumBarunpenR.ttf'

#color
WHITE = (255,255,255)
FOOD_COLOR = (0xff, 0x0,0x66)
SNAKE_COLOR = (0x66,0x66,0xff)
SNAKE_HEAD_COLOR = (0x33,0x33,0xff)
BACKGROUND_COLOR = (0x11,0x11,0x11)