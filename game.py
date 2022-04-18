import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math

pygame.init()

font = pygame.font.Font('monsterrat.tff', 50)
font_small = pygame.font.Font('monsterrat.tff', 30)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (223, 231, 239)
BLACK = (31, 39, 43)

BLOCK_SIZE = 40
SPEED = 100
OBSTACLE_MODE = True

class SnakeGameAI:

    def __init__(self, w=1280, h=960):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()
        self.block_size = BLOCK_SIZE
        self.obstacle_mode = OBSTACLE_MODE
        self.epsilon = 0
        self.game_num = 0
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w//2, self.h//2)
        self.snake = [self.head, Point(self.head.x-BLOCK_SIZE, self.head.y), Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.epsilon = 0
        self.game_num = 0
        self.food = None
        self.obstacles = []
        self.place_objects()
        self.frame_iteration = 0
    
    def place_objects(self, place_food=True, place_obstacle=True):
        if place_food:
            x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
            if Point(x, y) in self.snake or  Point(x, y) in self.obstacles:
                self.place_objects(True, False)
            else:
                self.food = Point(x, y)
        if place_obstacle and OBSTACLE_MODE:
            x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE     
            if Point(x, y) in self.snake or Point(x, y) in self.obstacles or Point(x, y) == self.food:
                self.place_objects(False, True)
            else:
                self.obstacles.append(Point(x, y))
            if self.score % 2 == 0:
                try:
                    self.obstacles.pop(0)
                except:
                    pass
    
    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self.move(action)
        self.snake.insert(0, self.head)
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_objects()
        else:
            self.snake.pop()

        self.update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        if pt in self.obstacles:
            return True
        return False

    def update_ui(self):
        self.display.fill(BLACK)
        for i in range(len(self.snake)):
            pygame.draw.rect(self.display, (121 + int(((132 - 121)//len(self.snake))*i), 192 + int(((237 - 192)//len(self.snake))*i), 255 - int(((255 - 193)//len(self.snake))*i)), pygame.Rect(self.snake[i].x, self.snake[i].y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.ellipse(self.display, (132, 237, 193), (self.food.x + 4, self.food.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8))
        for obstacle in self.obstacles:
            pygame.draw.polygon(self.display, color=(255, 80, 0), points=[(obstacle.x, obstacle.y + BLOCK_SIZE), (obstacle.x + (BLOCK_SIZE//2), obstacle.y), (obstacle.x + BLOCK_SIZE, obstacle.y + BLOCK_SIZE),])
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [20, 40])
        text = font_small.render("Game #" + str(self.game_num), True, WHITE)
        self.display.blit(text, [20, 105])
        text = font_small.render("Exploration Rate: " + "{:.1%}".format(self.epsilon), True, WHITE)
        self.display.blit(text, [20, 150])
        pygame.display.flip()

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_game_num(self, num):
        self.game_num = num
    
    def move(self, action):
        # action --> [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[index]
        elif np.array_equal(action, [0, 1, 0]):
            next_index = (index + 1) % 4
            new_dir = clock_wise[next_index]
        else:
            next_index = (index - 1) % 4
            new_dir = clock_wise[next_index]
        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        self.head = Point(x, y)
