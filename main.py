from time import sleep
from threading import Thread

from snakeObject import *
from NeuralNet import *
import numpy as np

snakes = []
wx = 21
wy = 20
block_size = 30
starting_deviation = 0




def initialize(amount):
    for num in range(amount):
        nn = NeuralNetwork(5, 15, 3)
        snakes.append(Snake(nn, wx, wy, block_size))


def run(generations):
    while generations > 0:
        for snake in snakes:
            snake.run()
        for snake in snakes:
            snake.nn = NeuralNetwork.combine_networks(choose_snake().nn, choose_snake().nn)
            snake.nn.mutate(1)
        print("Rounds Left:  " + str(generations))
        generations -= 1


def choose_snake():
    total_score = 0
    for snake in snakes:
        total_score += snake.score

    random_num = floor(random() * total_score)
    lower_bound = 0
    for snake in snakes:
        if lower_bound <= random_num < lower_bound + snake.score:
            return snake
        else:
            lower_bound += snake.score


def pick_best():
    best_score = 0
    for snake in snakes:
        if snake.score > best_score:
            best_score = snake.score
            best_snake = snake
    return best_snake


def calculate_deviation():
    total_ih_weights = 0
    total_ho_weights = 0
    total_h_biases = 0
    total_o_biases = 0
    for snake in snakes:
        total_ih_weights += snake.nn.ih_weights
        total_ho_weights += snake.nn.ho_weights
        total_h_biases += snake.nn.h_biases
        total_o_biases += snake.nn.o_biases
    average_ih_weights = total_ih_weights / len(snakes)
    average_ho_weights = total_ho_weights / len(snakes)
    average_h_biases = total_h_biases / len(snakes)
    average_o_biases = total_o_biases / len(snakes)

    deviation = 0
    for snake in snakes:
        deviation += np.sum((snake.nn.ih_weights - average_ih_weights))
        deviation += np.sum((snake.nn.ho_weights - average_ho_weights))
        deviation += np.sum((snake.nn.h_biases - average_h_biases))
        deviation += np.sum((snake.nn.o_biases - average_o_biases))

    return deviation

'''
class TrainingThread(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        initialize(70)
        sleep(2)
        print("Deviation of population before evolution is: %s" % calculate_deviation())
        run(4)
        print("Deviation of population after evolution is: %s" % calculate_deviation())
        print("Best score was: %s" % pick_best().score)
'''

initialize(200)
starting_deviation = calculate_deviation()
print("Deviation of population before evolution is: %s" % calculate_deviation())
pick_best().run(True)
run(10000)
print("Deviation of population after evolution is: %s" % calculate_deviation())
print("Best score was: %s" % pick_best().score)


