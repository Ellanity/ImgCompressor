# import statistics
import math
from copy import copy
from random import random
from MatrixCalculatorClass import MatrixCalculator


class NeuralNetwork:
    def __init__(self):
        self.layers = {}
        self.weights = {}
        self.calc = MatrixCalculator()

        self.max_allowable_error = 1
        self.total_error = 0
        self.quantity_of_iterations = 0
        self.delta = []

    ##### MICRO NETWORK FUNCTIONS

    class Layer:
        def __init__(self, layer_id: int, count_of_neurons: int, neural_network):
            self.id = layer_id
            self.count_of_neurons = count_of_neurons
            self.neurons = []
            self.neural_network = neural_network

        def addNeuron(self, value: float):
            self.neurons.append(self.Neuron(value))

        class Neuron:
            def __init__(self, value: float):
                self.value = value

        def __str__(self):
            string = ""
            for neuron in self.neurons:
                string += f"{neuron.value} "
            print(string)
            return ""

    def createLayer(self, layer_id: int, count_of_neurons: int):
        self.layers[layer_id] = self.Layer(layer_id, count_of_neurons, self)

    def fillLayer(self, layer_id: int, list_of_values: list):
        self.layers[layer_id].neurons = []
        for value in list_of_values:
            self.layers[layer_id].addNeuron(value)

    class MatrixOfWeights(MatrixCalculator.Matrix):
        def __init__(self, matrix_of_weights_id: str, layer_id_first: int, layer_id_second: int, neural_network):
            super().__init__()
            self.id = matrix_of_weights_id
            self.layer_id_first = layer_id_first
            self.layer_id_second = layer_id_second
            self.matrix = []
            self.neural_network = neural_network
            self.neurons_count_layer_first = self.neural_network.layers[self.layer_id_first].count_of_neurons
            self.neurons_count_layer_second = self.neural_network.layers[self.layer_id_second].count_of_neurons

            self.learning_rate = random() / 1000

        def createMatrix(self, list_of_weights: list):
            values = list_of_weights
            if len(list_of_weights) == 0:
                values = [random() for _ in range(self.neurons_count_layer_first * self.neurons_count_layer_second)]
            self.create(values, width=self.neurons_count_layer_second, height=self.neurons_count_layer_first)
            return self

    def createWeightsRandom(self, matrix_of_weights_id: str, layer_id_first: int, layer_id_second: int):
        self.weights[matrix_of_weights_id] = \
            self.MatrixOfWeights(matrix_of_weights_id, layer_id_first, layer_id_second, self).createMatrix([])

    def createWeights(self, matrix_of_weights_id: str, layer_id_first: int, layer_id_second: int, list_of_weights):
        self.weights[matrix_of_weights_id] = \
            self.MatrixOfWeights(matrix_of_weights_id, layer_id_first, layer_id_second, self).createMatrix(
                list_of_weights)

    ##### MACRO NETWORK FUNCTIONS

    def trainNeurons(self, layer_id_first: int, layer_id_second: int, matrix_of_weights_id: str):
        layer_first = self.layers[layer_id_first]
        matrix_of_weights = self.weights[matrix_of_weights_id]

        matrix_layer_first = self.calc.Matrix().create(elements=[neuron.value for neuron in layer_first.neurons],
                                                       width=layer_first.count_of_neurons, height=1)
        matrix_layer_second = self.calc.multiple(matrix_layer_first, matrix_of_weights)
        # print("MATRIX1\n", matrix_layer_first, "MATRIX2\n", matrix_of_weights, "MATRIX3\n", matrix_layer_second)
        self.fillLayer(layer_id=layer_id_second, list_of_values=matrix_layer_second.matrix[0])

    def getDeltaLayers(self, layer_id_first: int, layer_id_second: int):
        layer_first = self.layers[layer_id_first]
        layer_second = self.layers[layer_id_second]
        matrix_layer_first = self.calc.Matrix().create(elements=[neuron.value for neuron in layer_first.neurons],
                                                       width=layer_first.count_of_neurons, height=1)
        matrix_layer_second = self.calc.Matrix().create(elements=[neuron.value for neuron in layer_second.neurons],
                                                        width=layer_second.count_of_neurons, height=1)
        return self.calc.getListFromMatrix(self.calc.diff(matrix_layer_first, matrix_layer_second))

    def trainWeightsLast(self, matrix_of_weights_id: str, previous_layer_id: int):
        # W'(i+1) = W'(i) – α' * [Y(i)]T * dX(i)
        old_weights = copy(self.weights[matrix_of_weights_id])  # W'(i)
        delta_matrix = self.calc.Matrix()
        delta_matrix.create(elements=self.delta, width=len(self.delta), height=1)  # dX(i)
        previous_layer_matrix = self.calc.Matrix()
        previous_layer_matrix.create(elements=[neuron.value for neuron in self.layers[previous_layer_id].neurons],
                                     width=self.layers[previous_layer_id].count_of_neurons, height=1)
        deductible_p1 = self.calc.multiple(self.calc.transpose(previous_layer_matrix), delta_matrix)  # [Y(i)]T * dX(i)
        # α' * [Y(i)]T * dX(i) | W'(i)->a'
        deductible = self.calc.multipleNum(matrix=deductible_p1, num=self.weights[matrix_of_weights_id].learning_rate)
        new_weights = self.calc.diff(old_weights, deductible)  # W'(i) – α' * [Y(i)]T * dX(i)
        self.weights[matrix_of_weights_id].matrix = copy(new_weights.matrix)

    def trainWeights(self, matrix_of_weights_first_id: str, matrix_of_weights_second_id: str, previous_layer_id: int):
        # W(i+1) = W(i) – α * [X(i)]T * dX(i) * [W'(i)]T
        old_weights_first = copy(self.weights[matrix_of_weights_first_id])  # W(i)
        old_weights_second_transposed = self.calc.transpose(self.weights[matrix_of_weights_second_id])  # [W'(i)]T
        delta_matrix = self.calc.Matrix()
        delta_matrix.create(elements=self.delta, width=len(self.delta), height=1)  # dX(i)
        previous_layer = self.layers[previous_layer_id]
        previous_layer_matrix = self.calc.Matrix()
        previous_layer_matrix.create(elements=[neuron.value for neuron in previous_layer.neurons],
                                     width=previous_layer.count_of_neurons, height=1)  # X(i)
        deductible_p1 = self.calc.multiple(self.calc.transpose(previous_layer_matrix), delta_matrix)  # [X(i)]T * dX(i)
        deductible_p2 = self.calc.multiple(deductible_p1, old_weights_second_transposed)  # [X(i)]T * dX(i) * [W'(i)]T
        # α * [X(i)]T * dX(i) * [W'(i)]T  W(i)->a
        deductible = self.calc.multipleNum(matrix=deductible_p2,
                                           num=self.weights[matrix_of_weights_first_id].learning_rate)
        new_weights = self.calc.diff(old_weights_first, deductible)  # W(i) – α * [X(i)]T * dX(i) * [W'(i)]T
        self.weights[matrix_of_weights_first_id].matrix = copy(new_weights.matrix)

    def normalizeWeights(self, matrix_of_weights_id: str):
        # transposed_matrix = self.calc.transpose(self.weights[matrix_of_weights_id])
        for i in range(0, self.weights[matrix_of_weights_id].height):
            sqsum = 0
            for k in self.weights[matrix_of_weights_id].matrix[i]:
                sqsum += k * k
            module = math.sqrt(sqsum)
            for j in range(0, self.weights[matrix_of_weights_id].width):
                """sqsum = 0
                for k in transposed_matrix.matrix[j]:
                    sqsum += k * k
                module = math.sqrt(sqsum)"""
                self.weights[matrix_of_weights_id].matrix[i][j] /= module

    def calculateTotalRMSError(self):
        # Е(q) = E(dX(q)i * dX(q)i), where 1<=i<=N
        RMS = 0
        for value in self.delta:
            RMS += value * value
        return RMS

    ##### GLOBAL NETWORK FUNCTIONS

    def trainNeuronNetwork(self, blocks):
        self.creatingNetwork(blocks=blocks)
        while self.total_error > self.max_allowable_error or self.total_error == 0:
            for block in blocks:
                self.trainingIteration(block)
                """print(f"ITERATION: {self.quantity_of_iterations}"
                     f"\t errors sum: {self.total_error:.{10}f}"
                     f"\t max_error: {self.max_allowable_error}"
                     # f"\t current error: {error}"
                     f"\t average error: {(self.total_error / len(blocks)):.{4}f}"
                     f"\t ok block: {(self.total_error / len(blocks)) < self.max_allowable_error}")"""
            if (self.total_error / len(blocks)) < self.max_allowable_error or \
                    self.quantity_of_iterations > 100:
                break
            self.total_error = 0

    def creatingNetwork(self, blocks):
        # create layers
        block_length = len(blocks[0])
        print("creating layer 1")
        self.createLayer(layer_id=1, count_of_neurons=block_length)
        print("creating layer 2")
        self.createLayer(layer_id=2, count_of_neurons=int(block_length * 1.5))
        print("creating layer 3")
        self.createLayer(layer_id=3, count_of_neurons=block_length)
        # crete matrices of weights 1-2/2-3
        print("creating weights matrix 1-2")
        self.createWeightsRandom(matrix_of_weights_id="1-2", layer_id_first=1, layer_id_second=2)
        weights = self.calc.getListFromMatrix(self.calc.transpose(self.weights["1-2"]))
        print("creating weights matrix 2-3")
        self.createWeights(matrix_of_weights_id="2-3", layer_id_first=2, layer_id_second=3, list_of_weights=weights)

        print("creating finished")
        self.max_allowable_error *= block_length

    def trainingIteration(self, block):
        self.fillLayer(layer_id=1, list_of_values=block)
        self.trainNeurons(layer_id_first=1, layer_id_second=2, matrix_of_weights_id="1-2")  # training 2 layer
        self.trainNeurons(layer_id_first=2, layer_id_second=3, matrix_of_weights_id="2-3")  # training 3 layer
        self.delta = self.getDeltaLayers(1, 3)
        self.trainWeights(matrix_of_weights_first_id="1-2", matrix_of_weights_second_id="2-3", previous_layer_id=1)
        self.trainWeightsLast(matrix_of_weights_id="2-3", previous_layer_id=2)

        self.normalizeWeights("1-2")
        self.normalizeWeights("2-3")

        self.total_error += self.calculateTotalRMSError()
        self.quantity_of_iterations += 1

    def runBlockThroughNN(self, block):
        self.fillLayer(layer_id=1, list_of_values=block)
        self.trainNeurons(layer_id_first=1, layer_id_second=2, matrix_of_weights_id="1-2")  # training 2 layer
        self.trainNeurons(layer_id_first=2, layer_id_second=3, matrix_of_weights_id="2-3")  # training 3 layer
        return [neuron.value for neuron in self.layers[3].neurons]
