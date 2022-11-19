import copy
import math
from random import randint
# import numpy as np
from MatrixClass import Matrix


class NN:
    def __init__(self):
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.weights1 = None
        self.weights2 = None
        self.totalRMS = None
        self.maxRMS = None
        self.learning_rate = None
        self.coeff_of_quantity_of_neurons_second_layer = None
        self.otherVariables = None

    def createNN(self, variables):
        self.otherVariables = variables
        if self.otherVariables is None:
            return
        self.totalRMS = 0
        self.coeff_of_quantity_of_neurons_second_layer = self.otherVariables["coeff_of_quantity_of_neurons_second_layer"]
        block_size = self.otherVariables["block_width"] * self.otherVariables["block_height"]
        self.maxRMS = 0.1 * self.coeff_of_quantity_of_neurons_second_layer * block_size * 4
        self.learning_rate = self.otherVariables["learning_rate"]
        # create weights matrices
        elements = []
        for _ in range(block_size * block_size * self.coeff_of_quantity_of_neurons_second_layer):
            elements.append(randint(int(-1e5), int(1e5)) / 1e5)
        self.weights1 = Matrix(width=block_size * self.coeff_of_quantity_of_neurons_second_layer, height=block_size, elements=elements)
        self.weights2 = self.weights1.transpose()

    def trainNN(self, blocks):
        train_iteration_num = 0
        block_size = len(blocks[0])

        while self.totalRMS > self.maxRMS or self.totalRMS == 0:
            self.totalRMS = 0
            for block_index in range(len(blocks)):
                self.layer1 = Matrix(width=block_size, height=1, elements=blocks[block_index])
                self.layer2 = self.layer1 * self.weights1
                self.layer3 = self.layer2 * self.weights2
                delta_layers13 = self.layer3 - self.layer1
                rms = sum([value * value for value in delta_layers13.getList()])
                self.totalRMS += rms

                new_weights1 = self.weights1 - (self.layer1.transpose() * delta_layers13 * self.weights2.transpose()) * self.learning_rate
                new_weights2 = self.weights2 - (self.layer2.transpose() * delta_layers13) * self.learning_rate
                #print("old-weights1:\n", self.weights1, "old-weights2:\n", self.weights2)
                self.weights1 = new_weights1
                self.weights2 = new_weights2
                self.normalizeWeights(self.weights1)
                self.normalizeWeights(self.weights2)
                #print("new-weights1:\n", self.weights1, "new-weights2:\n", self.weights2)
                train_iteration_num += 1

                print("IT", train_iteration_num, "RMS:", self.totalRMS, "MAX", self.maxRMS)

        # print("layer1: ",self.layer1, "layer2: ", self.layer2, "layer3: ", self.layer3)
        # print("weights1:\n", self.weights1, "weights2:\n", self.weights2)

    def normalizeWeights(self, weights):
        for i in range(0, weights.height):
            sqsum = 0
            for k in weights.matrix[i]:
                sqsum += k * k
            module = math.sqrt(sqsum)
            for j in range(0, weights.width):
                weights.matrix[i][j] /= module

    def NNIteration(self, block):
        self.layer1 = Matrix(width=len(block), height=1, elements=block)
        self.layer2 = self.layer1 * self.weights1
        self.layer3 = self.layer2 * self.weights2
        for index in range(len(self.layer3.matrix[0])):
            if self.layer3.matrix[0][index] < -1:
                self.layer3.matrix[0][index] = -1
            if self.layer3.matrix[0][index] > 1:
                self.layer3.matrix[0][index] = 1

        return copy.deepcopy(self.layer3.getList())

    def saveWeights(self):
        print(self.otherVariables["image_name"],
              self.otherVariables["channel_id"],
              self.otherVariables["block_width"],
              self.otherVariables["block_height"])
