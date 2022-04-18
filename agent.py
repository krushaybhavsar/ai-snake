from numpy.core.fromnumeric import mean
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 # learning rate
MAX_RAND_GAMES = 50 # max number of games played with a exploration varient [50, 100, 150]

class Agent:

    def __init__(self):
        self.n_games = 0 # number of games played
        self.epsilon = 0 # exploration rate (randomness)
        self.gamma = 0.9 # discount rate (value smaller than 1)
        self.memory = deque(maxlen=MAX_MEMORY) # memory of experiences; calls popleft()
        self.model = Linear_QNet(11, 256, 3) # neural network with 11 inputs, 3 outputs, 1 hidden layer with 256 neurons
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) # trainer for neural network

    def get_state(self, game):
        head = game.snake[0]
        # get the points around the head of the snake
        point_l = Point(head.x - game.block_size, head.y)
        point_r = Point(head.x + game.block_size, head.y)
        point_u = Point(head.x, head.y - game.block_size)
        point_d = Point(head.x, head.y + game.block_size)
        # get the direction of the snake
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        # state = [danger_straight, danger_right, danger_left,
        #           direction_left, direction_right, direction_up, direction_down, 
        #           food_left, food_right, food_up, food_down]
        state = [
            # danger_straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            # danger_right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            # danger_left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            # snake move direction
            dir_l, # direction_left
            dir_r, # direction_right
            dir_u, # direction_up
            dir_d, # direction_down
            # food direction
            game.food.x < game.head.x, # food_left
            game.food.x > game.head.x, # food_right
            game.food.y < game.head.y, # food_up
            game.food.y > game.head.y # food_down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = MAX_RAND_GAMES - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            idx = random.randint(0, 2)
            final_move[idx] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)
        game.set_game_num(agent.n_games)
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()
            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            if agent.epsilon > 0:
                game.set_epsilon(agent.epsilon/200)
            else:
                game.set_epsilon(0)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            print('Mean Score:', mean_score)
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()