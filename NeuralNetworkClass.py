# import statistics
import math
from copy import copy
from random import random
from MatrixClass import Matrix


class NeuralNetwork:
    def __init__(self):
        self.layers = {}
        self.weights = {}
        self.delta = []

        self.max_allowable_error = 5
        self.total_error = 0
        self.quantity_of_iterations = 0

    ##### MICRO NETWORK FUNCTIONS

    class Layer:
        def __init__(self, layer_id: int, count_of_neurons: int, neural_network):
            self.neural_network = neural_network
            self.id = layer_id
            self.count_of_neurons = count_of_neurons
            self.neurons = []

        def addNeuron(self, value: float):
            self.neurons.append(self.Neuron(value))

        class Neuron:
            def __init__(self, value: float):
                self.value = value
                if self.value > 1:
                    self.value = 1
                if self.value < -1:
                    self.value = -1

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

    class MatrixOfWeights(Matrix):
        def __init__(self, neural_network, matrix_of_weights_id: str, layer_id_first: int, layer_id_second: int, list_of_weights: list):
            self.neural_network = neural_network
            self.id = matrix_of_weights_id
            self.matrix = []
            self.layer_id_first = layer_id_first
            self.layer_id_second = layer_id_second
            self.neurons_count_layer_first = self.neural_network.layers[layer_id_first].count_of_neurons
            self.neurons_count_layer_second = self.neural_network.layers[layer_id_second].count_of_neurons
            self.learning_rate = 0.001  # random() / 100

            elements = list_of_weights
            if len(elements) == 0:
                elements = [random() for _ in range(self.neurons_count_layer_first * self.neurons_count_layer_second)]
            super().__init__(width=self.neurons_count_layer_second, height=self.neurons_count_layer_first, elements=elements)

    ##### MACRO NETWORK FUNCTIONS

    def trainNeurons(self, layer_id_first: int, layer_id_second: int, matrix_of_weights_id: str):
        layer_first = self.layers[layer_id_first]
        matrix_of_weights = self.weights[matrix_of_weights_id]

        matrix_layer_first = Matrix(width=layer_first.count_of_neurons, height=1,
                                    elements=[neuron.value for neuron in layer_first.neurons])
        matrix_layer_second = matrix_layer_first * matrix_of_weights
        # print("trainNeurons:", layer_id_first, layer_id_second, matrix_layer_first, matrix_layer_second, matrix_of_weights)
        self.fillLayer(layer_id=layer_id_second, list_of_values=matrix_layer_second.getList())

    def getDeltaLayers(self, layer_id_first: int, layer_id_second: int):
        layer_first = self.layers[layer_id_first]
        layer_second = self.layers[layer_id_second]
        matrix_layer_first = Matrix(width=layer_first.count_of_neurons, height=1,
                                    elements=[neuron.value for neuron in layer_first.neurons])
        matrix_layer_second = Matrix(width=layer_second.count_of_neurons, height=1,
                                     elements=[neuron.value for neuron in layer_second.neurons])
        return matrix_layer_first - matrix_layer_second

    def trainWeightsLast(self, matrix_of_weights_id: str, previous_layer_id: int):
        # W'(i+1) = W'(i) – α' * [Y(i)]T * dX(i)
        # W'(i)
        old_weights = copy(self.weights[matrix_of_weights_id])
        # Y(i)
        previous_layer_matrix = Matrix(width=self.layers[previous_layer_id].count_of_neurons, height=1,
                                       elements=[neuron.value for neuron in self.layers[previous_layer_id].neurons])
        # [Y(i)]T * dX(i) * α'
        deductible = previous_layer_matrix.transpose() * self.delta * self.weights[matrix_of_weights_id].learning_rate
        # W'(i) – [Y(i)]T * dX(i) * a'
        new_weights = old_weights - deductible
        # W'(i+1) = W'(i) – [Y(i)]T * dX(i) * a'
        self.weights[matrix_of_weights_id].matrix = copy(new_weights.matrix)

    def trainWeights(self, matrix_of_weights_first_id: str, matrix_of_weights_second_id: str, previous_layer_id: int):
        # W(i+1) = W(i) – α * [X(i)]T * dX(i) * [W'(i)]T
        previous_layer = self.layers[previous_layer_id]

        # W(i)
        old_weights_first = copy(self.weights[matrix_of_weights_first_id])
        # [W'(i)]T
        old_weights_second_transposed = self.weights[matrix_of_weights_second_id].transpose()
        # X(i)
        previous_layer_matrix = Matrix(width=previous_layer.count_of_neurons, height=1,
                                       elements=[neuron.value for neuron in previous_layer.neurons])
        # [X(i)]T * dX(i) * [W'(i)]T * a'
        deductible = previous_layer_matrix.transpose() * self.delta * old_weights_second_transposed * self.weights[matrix_of_weights_first_id].learning_rate
        # W(i+1) = W(i) – α * [X(i)]T * dX(i) * [W'(i)]T
        new_weights = old_weights_first - deductible
        self.weights[matrix_of_weights_first_id].matrix = copy(new_weights.matrix)

    def normalizeWeights(self, matrix_of_weights_id: str):
        for i in range(0, self.weights[matrix_of_weights_id].height):
            sqsum = 0
            for k in self.weights[matrix_of_weights_id].matrix[i]:
                sqsum += k * k
            module = math.sqrt(sqsum)
            for j in range(0, self.weights[matrix_of_weights_id].width):
                self.weights[matrix_of_weights_id].matrix[i][j] /= module

    def normalizeAllWeights(self):
        for weights in self.weights.keys():
            self.normalizeWeights(weights)

    def calculateTotalRMSError(self):
        # Е(q) = E(dX(q)i * dX(q)i), where 1<=i<=N
        RMS = 0
        for value in self.delta.getList():
            RMS += value * value
        return RMS

    ##### GLOBAL NETWORK FUNCTIONS

    def trainNeuronNetwork(self, blocks):
        self.creatingNetwork(blocks=blocks)
        print("training start")
        while self.total_error > self.max_allowable_error or self.total_error == 0:
            for block in blocks:
                self.trainingIteration(block)
                """print(f"ITERATION: {self.quantity_of_iterations}"
                      f"\t errors sum: {self.total_error:.{10}f}"
                      f"\t max_error: {self.max_allowable_error}"
                      # f"\t current error: {error}"
                      f"\t average error: {(self.total_error / len(blocks)):.{4}f}"
                      f"\t ok block: {(self.total_error / len(blocks)) < self.max_allowable_error}")"""
            if ((self.total_error / len(blocks)) < self.max_allowable_error and self.quantity_of_iterations > len(blocks) * 3) or self.quantity_of_iterations > len(blocks) * 5:
                break
            self.total_error = 0
        print("training finished")

    def creatingNetwork(self, blocks):
        # create layers
        print("creating start")
        block_length = len(blocks[0])
        self.createLayer(layer_id=1, count_of_neurons=block_length)
        self.createLayer(layer_id=2, count_of_neurons=math.ceil(block_length * 2)) #0.8))
        self.createLayer(layer_id=3, count_of_neurons=block_length)
        # crete matrices of weights 1-2
        self.weights["1-2"] = self.MatrixOfWeights(neural_network=self, matrix_of_weights_id="1-2",
                                                   layer_id_first=1, layer_id_second=2, list_of_weights=[])
        # crete matrices of weights 2-3
        list_of_weights = self.weights["1-2"].transpose().getList()
        self.weights["2-3"] = self.MatrixOfWeights(neural_network=self, matrix_of_weights_id="2-3",
                                                   layer_id_first=2, layer_id_second=3, list_of_weights=list_of_weights)
        print("creating finished")

    def trainingIteration(self, block):
        self.fillLayer(layer_id=1, list_of_values=block)
        self.trainNeurons(layer_id_first=1, layer_id_second=2, matrix_of_weights_id="1-2")  # training 2 layer
        self.trainNeurons(layer_id_first=2, layer_id_second=3, matrix_of_weights_id="2-3")  # training 3 layer
        self.delta = self.getDeltaLayers(1, 3)
        self.trainWeights(matrix_of_weights_first_id="1-2", matrix_of_weights_second_id="2-3", previous_layer_id=1)
        self.trainWeightsLast(matrix_of_weights_id="2-3", previous_layer_id=2)
        self.normalizeAllWeights()

        rms = self.calculateTotalRMSError()
        # print(rms)
        self.total_error += rms
        self.quantity_of_iterations += 1

    def runBlockThroughNeuralNetwork(self, block):
        self.fillLayer(layer_id=1, list_of_values=block)
        self.trainNeurons(layer_id_first=1, layer_id_second=2, matrix_of_weights_id="1-2")  # training 2 layer
        self.trainNeurons(layer_id_first=2, layer_id_second=3, matrix_of_weights_id="2-3")  # training 3 layer
        #block_new1 = [neuron.value for neuron in self.layers[1].neurons]
        #block_new2 = [neuron.value for neuron in self.layers[2].neurons]
        block_new3 = [neuron.value for neuron in self.layers[3].neurons]
        #print("layers:", block_new1, block_new2, block_new3)
        #print("weights 1:", self.weights["1-2"], "weights 2:", self.weights["2-3"])
        return block_new3
