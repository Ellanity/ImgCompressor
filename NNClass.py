import copy
import math
from pathlib import Path
import os.path
from random import randint
from MatrixClass import Matrix


class NN:
    def __init__(self):
        self.layer_1: Matrix = None
        self.layer_2: Matrix = None
        self.layer_3: Matrix = None
        self.weights_1: Matrix = None
        self.weights_2: Matrix = None

        self.otherVariables: dict = {}
        self.block_size: int = 0
        self.compression_rate: float = 2
        self.learning_rate: float = 0
        self.maxRMS: float = 0
        self.totalRMS: float = 0
        self.train_iteration_num: int = 0
        self.loaded_weights: bool = False
        self.can_load_weights: bool = True

    def createNN(self, variables):
        self.otherVariables = variables
        if self.otherVariables == {}:
            return
        # Set variables to create network
        self.compression_rate = self.otherVariables["compression_rate"]
        self.block_size = self.otherVariables["block_width"] * self.otherVariables["block_height"]
        # Correct formula for calculating maxRMS (use [* 4]s to speed up tests):
        # self.maxRMS = 0.1 * self.compression_rate * self.block_size
        self.maxRMS = 0.1 * self.compression_rate * self.block_size * 4
        self.can_load_weights = self.otherVariables["can_load_weights"]
        self.learning_rate = self.otherVariables["learning_rate"]
        # Create weights matrices
        elements = []
        for _ in range(int(self.block_size * math.ceil(self.block_size * self.compression_rate))):
            elements.append(randint(int(-1e5), int(1e5)) / 1e5)

        if self.can_load_weights:
            self.__loadWeights()
        if not self.loaded_weights:
            self.weights_1 = Matrix(width=math.ceil(self.block_size * self.compression_rate),
                                    height=self.block_size, elements=elements)
            self.weights_2 = self.weights_1.transpose()
        print("Neural network created successfully")

    def trainNN(self, blocks):
        block_size = len(blocks[0])

        if self.loaded_weights:
            coeff_Z = (self.block_size * len(blocks)) / \
                      ((self.block_size + len(blocks)) * self.block_size * self.compression_rate + 2)
            print(f"Compression ratio: {coeff_Z}")
            return
        while self.totalRMS > self.maxRMS or (self.totalRMS == 0 and self.train_iteration_num == 0):
            self.totalRMS = 0
            for block_index in range(len(blocks)):
                # Create layers
                self.layer_1 = Matrix(width=block_size, height=1, elements=blocks[block_index])
                self.layer_2 = self.layer_1 * self.weights_1
                self.layer_3 = self.layer_2 * self.weights_2
                # Calculate rms and delta between first and last layer
                delta_layers13: Matrix = self.layer_3 - self.layer_1
                rms = sum([value * value for value in delta_layers13.getList()])
                self.totalRMS += rms
                # Update weights
                new_weights1 = self.weights_1 - (self.layer_1.transpose() * delta_layers13 * self.weights_2.transpose()) * self.learning_rate
                new_weights2 = self.weights_2 - (self.layer_2.transpose() * delta_layers13) * self.learning_rate
                self.weights_1 = new_weights1
                self.weights_2 = new_weights2
                self.normalizeWeights(self.weights_1)
                self.normalizeWeights(self.weights_2)
                # Output
                self.train_iteration_num += 1
                print(f"\rmax error: {self.maxRMS:.{5}f} | "
                      f"current error: {self.totalRMS:.{5}f} | "
                      f"Iterations: {self.train_iteration_num}", end="")
        print("\r")
        coeff_Z = (self.block_size * len(blocks)) /\
                  ((self.block_size + len(blocks)) * self.block_size * self.compression_rate + 2)
        print(f"Compression ratio: {coeff_Z}")
        self.__saveWeights()

    def normalizeWeights(self, weights):
        for i in range(0, weights.height):
            sum_of_squares = 0
            for k in weights.matrix[i]:
                sum_of_squares += k * k
            module = math.sqrt(sum_of_squares)
            for j in range(0, weights.width):
                weights.matrix[i][j] /= module

    def NNIteration(self, block):
        self.layer_1 = Matrix(width=len(block), height=1, elements=block)
        self.layer_2 = self.layer_1 * self.weights_1
        self.layer_3 = self.layer_2 * self.weights_2

        """
        This function is used to get the final result of the function.
        Therefore, if numbers appear outside the range from -1 to 1,
        the value will be converted incorrectly and a prominent artifact
        will appear in the picture. 
        To avoid this, we cut off all values that go beyond the limits.
        """
        for index in range(len(self.layer_3.matrix[0])):
            self.layer_3.matrix[0][index] = -1 if self.layer_3.matrix[0][index] < -1 else self.layer_3.matrix[0][index]
            self.layer_3.matrix[0][index] = 1 if self.layer_3.matrix[0][index] > 1 else self.layer_3.matrix[0][index]

        return {
            "compressed": copy.deepcopy(self.layer_2.getList()),
            "final": copy.deepcopy(self.layer_3.getList())
        }

    def getWeightsFileName(self, weights_index):
        try:
            image_name = Path(self.otherVariables['image_name']).stem
            return f"weights/{image_name}/" \
                   f"{self.otherVariables['image_width']}_" \
                   f"{self.otherVariables['image_height']}_" \
                   f"{self.otherVariables['count_of_channels']}_" \
                   f"{self.otherVariables['block_width']}_" \
                   f"{self.otherVariables['block_height']}_" \
                   f"{self.otherVariables['compression_rate']}_" \
                   f"{self.otherVariables['channel_id']}_" \
                   f"{weights_index}.nnwght"
        except Exception as ex:
            print(ex)

    def __saveWeights(self):
        try:
            image_name = Path(self.otherVariables['image_name']).stem
            Path(f"weights/{image_name}").mkdir(parents=True, exist_ok=True)
            with open(self.getWeightsFileName(weights_index=1), "w") as file:
                for row in self.weights_1.matrix:
                    string = ""
                    for x in row:
                        string += str(x) + " "
                    file.write(string + "\n")
            with open(self.getWeightsFileName(weights_index=2), "w") as file:
                for row in self.weights_2.matrix:
                    string = ""
                    for x in row:
                        string += str(x) + " "
                    file.write(string + "\n")

        except Exception as ex:
            print(ex)

    def __loadWeights(self):
        if self.can_load_weights is not True:
            print("Weights will not be loaded")
            return
        try:
            file_name_1 = self.getWeightsFileName(weights_index=1)
            file_name_2 = self.getWeightsFileName(weights_index=2)
            if os.path.exists(file_name_1) and os.path.exists(file_name_2):
                with open(file_name_1, "r") as file:
                    elements = []
                    height = 0
                    for line in file.readlines():
                        for x in line[:-2].split(' '):
                            elements.append(float(x))
                        height += 1
                    self.weights_1 = Matrix(width=int(len(elements) / height), height=height, elements=elements)
                with open(file_name_2, "r") as file:
                    elements = []
                    height = 0
                    for line in file.readlines():
                        for x in line[:-2].split(' '):
                            elements.append(float(x))
                        height += 1
                    self.weights_2 = Matrix(width=int(len(elements) / height), height=height, elements=elements)
                self.loaded_weights = True
            else:
                print("No weights for this parameters")
        except Exception as ex:
            print(ex)
