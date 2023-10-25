from random import random, uniform

import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


matrix_sig = np.vectorize(sigmoid)


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


matrix_dsig = np.vectorize(dsigmoid)


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.ih_weights = np.random.randn(input_nodes, hidden_nodes)
        self.ho_weights = np.random.randn(hidden_nodes, output_nodes)

        self.h_biases = np.random.randn(hidden_nodes, 1)
        self.o_biases = np.random.randn(output_nodes, 1)

        self.loss = 0

    # Input_vector should be an array
    def run(self, input_array):
        input_vector = np.array([input_array]).transpose()
        if input_vector.size != self.input_nodes:
            print("Incorrect input vector size")

        ih_weights_transposed = np.transpose(self.ih_weights)
        ho_weights_transposed = np.transpose(self.ho_weights)

        # Here I am using 3blue1brown convention
        hidden_z = np.add(np.dot(ih_weights_transposed, input_vector), self.h_biases)
        hidden_a = matrix_sig(hidden_z)

        output_z = np.add((ho_weights_transposed @ hidden_a), self.o_biases)
        output_a = matrix_sig(output_z)

        return output_a.flatten().tolist()

    # training data should be array of input output tuples(each input and output should be an array)
    def train(self, training_data, cycles):
        step = 0.5
        for cycle in range(cycles):
            random_sample = math.floor(random() * len(training_data))
            training_input_array = training_data[random_sample][0]
            training_output_array = training_data[random_sample][1]

            training_input = np.array([training_input_array]).transpose()
            training_output = np.array([training_output_array]).transpose()

            ih_weights_transposed = np.transpose(self.ih_weights)
            ho_weights_transposed = np.transpose(self.ho_weights)

            # Here I am using 3blue1brown convention
            hidden_z = np.add((ih_weights_transposed @ training_input), self.h_biases)
            hidden_a = matrix_sig(hidden_z)

            output_z = np.add((ho_weights_transposed @ hidden_a), self.o_biases)
            actual_output = matrix_sig(output_z)

            # Now that everything has been forward propagated, we need to calculate the partial derivatives
            error = actual_output - training_output

            o_baises_gradient = error * matrix_dsig(output_z)

            hidden_a_expansion = np.repeat(hidden_a, self.output_nodes, 1)
            cost_in_terms_z = (error * matrix_dsig(output_z)).transpose()
            cost_in_terms_z_expansion = np.repeat(cost_in_terms_z, self.hidden_nodes, 0)
            ho_weights_gradient = cost_in_terms_z_expansion * hidden_a_expansion

            hidden_partial_derivative = self.ho_weights @ (error * matrix_dsig(output_z))

            h_baises_gradient = hidden_partial_derivative * matrix_dsig(hidden_z)

            training_input_expansion = np.repeat(training_input, self.hidden_nodes, 1)
            hidden_cost_in_terms_z = (hidden_partial_derivative * matrix_dsig(hidden_z)).transpose()
            hidden_cost_in_terms_z_expansion = np.repeat(hidden_cost_in_terms_z, self.input_nodes, 0)
            ih_weights_gradient = hidden_cost_in_terms_z_expansion * training_input_expansion

            # Change the weights and biases
            self.h_biases = self.h_biases - (step * h_baises_gradient)
            self.o_biases = self.o_biases - (step * o_baises_gradient)

            self.ih_weights = self.ih_weights - (step * ih_weights_gradient)
            self.ho_weights = self.ho_weights - (step * ho_weights_gradient)

    def run_and_choose(self, input):
        output_array = self.run(input)
        best_result = 0
        for number in range(len(output_array)):
            if output_array[number] > output_array[best_result]:
                best_result = number

        return best_result


    def percent_correct(self, ta):
        amount_correct = 0
        total_samples = 0
        for sample in ta:
            for number in range(len(sample[1])):
                if sample[1][number] == 1:
                    correct_num = number
            if self.run_and_choose(sample[0]) == correct_num:
                amount_correct = amount_correct + 1
            total_samples = total_samples + 1
        return amount_correct / total_samples

    def loss(self, ta):
        total_loss = 0
        for sample in ta:
            output = self.run(sample[0])
            target = np.array(sample[1]).reshape(len(sample[1]), 1)
            error_vector = output - target
            loss = np.dot(error_vector.transpose(), error_vector)
            total_loss += loss
        return total_loss.item()

    @staticmethod
    def combine_networks(n1, n2):
        new_net = NeuralNetwork(n1.input_nodes, n1.hidden_nodes, n1.output_nodes)
        o_break = math.floor(random() * n1.output_nodes)
        h_break = math.floor(random() * n1.hidden_nodes)

        n1_o_biases = n1.o_biases[0:o_break]
        n1_ho_weights = n1.ho_weights[0:o_break]
        n1_h_biases = n1.h_biases[0:h_break]
        n1_ih_weights = n1.ih_weights[0:h_break]

        n2_o_biases = n2.o_biases[o_break:]
        n2_ho_weights = n2.ho_weights[o_break:]
        n2_h_biases = n2.h_biases[h_break:]
        n2_ih_weights = n2.ih_weights[h_break:]

        new_net.ih_weights = np.concatenate((n1_ih_weights, n2_ih_weights), axis=0)
        new_net.ho_weights = np.concatenate((n1_ho_weights, n2_ho_weights), axis=0)
        new_net.h_biases = np.concatenate((n1_h_biases, n2_h_biases), axis=0)
        new_net.o_biases = np.concatenate((n1_o_biases, n2_o_biases), axis=0)

        return new_net

    def mutate(self, percent):
        random_num = math.floor(random() * 100 / percent)
        if random_num == 1:
            # Change HO Weights
            random_num_i = math.floor(random() * self.hidden_nodes)
            random_num_j = math.floor(random() * self.output_nodes)
            self.ho_weights[random_num_i][random_num_j] = uniform(-1, 1)
            # Change IG Weights
            random_num_i = math.floor(random() * self.input_nodes)
            random_num_j = math.floor(random() * self.hidden_nodes)
            self.ih_weights[random_num_i][ random_num_j] = uniform(-1, 1)

            #Choose and change a bias
            random_num = math.floor(random() * 2)
            if random_num == 0:
                #Output bias
                random_num = math.floor(random() * self.output_nodes)
                self.o_biases[random_num] = uniform(-1, 1)
            else:
                random_num = math.floor(random() * self.hidden_nodes)
                self.h_biases[random_num] = uniform(-1, 1)



















