from math import floor
from random import random
import pygame


class Snake:

    def __init__(self, nn, wx, wy, block_size):
        self.wx = wx
        self.wy = wy
        self.block_size = block_size
        self.color = (255, 0, 0)
        self.nn = nn
        self.score = 4

    def run(self, paint=False):
        pygame.init()
        pygame.fastevent.init()
        if paint:
            pygame.init()
            pygame.fastevent.init()
            win = pygame.display.set_mode((630, 600))
            pygame.display.set_caption("Best Snake")

        def pick_apple():
            apple = [floor(random() * self.wx), floor(random() * self.wy)]
            for block in snake:
                if apple == block:
                    pick_apple()
            return apple

        def process_vector(array):
            biggest_value = 0
            location = 0
            for n in range(len(array)):
                if array[n] > biggest_value:
                    biggest_value = array[n]
                    location = n
            new_array = [0] * len(array)
            new_array[location] = 1

            return new_array

        def pick(array):
            biggest_value = 0
            location = 0
            for n in range(len(array)):
                if array[n] > biggest_value:
                    biggest_value = array[n]
                    location = n

            return location

        moves_since_apple = 0
        run = True
        score = 4
        snake = [
            [7, 10],
            [7 - 1, 10],
            [7 - 2, 10],
            [7 - 3, 10]
        ]
        apple = [9, 10]
        direction = 1
        while run:
            # First value:How far ahead apple is
            # Second Value: How for to the right apple is
            # Third Value: distance to obsticle to the left
            # Fourth Value: distance to obsticle ahead
            # Fifth Value: distance to obsticle to the right
            if paint:
                pygame.fastevent.get()
                pygame.time.delay(120)
                win.fill((0, 0, 0))

            input_vector = [0, 0, 0, 0, 0]
            apple_options = [snake[0][1] - apple[1], apple[0] - snake[0][0], apple[1] - snake[0][1],
                             snake[0][0] - apple[0]]
            options = [snake[0][1], self.wx - snake[0][0], self.wy - snake[0][1], snake[0][0]]

            for block in snake[1:]:
                # change north boundry
                if block[0] == snake[0][0] and block[1] < snake[0][1]:
                    if snake[0][1] - block[1] < options[0]:
                        options[0] = snake[0][1] - block[1]
                # Change south boundry
                if block[0] == snake[0][0] and block[1] > snake[0][1]:
                    if block[1] - snake[0][1] < options[2]:
                        options[2] = block[1] - snake[0][1]
                # Change east boundry
                if block[1] == snake[0][1] and block[0] > snake[0][0]:
                    if block[0] - snake[0][0] < options[1]:
                        options[1] = block[0] - snake[0][0]
                if block[1] == snake[0][1] and block[0] < snake[0][0]:
                    if snake[0][0] - block[0] < options[3]:
                        options[3] = snake[0][0] - block[0]

            input_vector[0] = apple_options[direction] / self.wx * 2
            input_vector[1] = apple_options[(direction + 1) % 4] / self.wx * 2
            input_vector[2] = options[-1 + direction] / self.wx
            input_vector[3] = options[direction] / self.wx
            input_vector[4] = options[(direction + 1) % 4] / self.wx

            output_vector = self.nn.run(input_vector)
            output = pick(output_vector) - 1

            # output -1 is left 0 is dont change and 1 is right(all relative to snake head)
            snake = snake[:-1]
            new_head_locations = [[snake[0][0], snake[0][1] - 1], [snake[0][0] + 1, snake[0][1]],
                                  [snake[0][0], snake[0][1] + 1], [snake[0][0] - 1, snake[0][1]]]
            b = output + direction
            if b == 4:
                b = 0
            if b == -1:
                b = 3
            snake.insert(0, new_head_locations[b])
            direction = b

            if snake[0][0] >= self.wx or snake[0][0] < 0 or snake[0][1] >= self.wy or snake[0][1] < 0 or snake[
                0] in snake[1:]:
                run = False

            if moves_since_apple > 55:
                run = False

            if snake[0] == apple:
                snake.insert(len(snake) - 1, apple)
                apple = pick_apple()
                score = score + 1
                moves_since_apple = 0
            else:
                moves_since_apple += 1

            if paint:
                for block in snake[1:]:
                    pygame.draw.rect(win, self.color, (
                    block[0] * self.block_size, block[1] * self.block_size, self.block_size - 4, self.block_size - 4))
                pygame.draw.rect(win, (0, 255, 255), (
                snake[0][0] * self.block_size, snake[0][1] * self.block_size, self.block_size - 4, self.block_size - 4))
                pygame.draw.rect(win, (255, 255, 255), (
                apple[0] * self.block_size, apple[1] * self.block_size, self.block_size - 1, self.block_size - 1))
                pygame.display.update()
        pygame.quit()
        self.score = score
