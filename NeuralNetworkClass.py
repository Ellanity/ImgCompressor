from copy import copy
from random import random
from MatrixCalculatorClass import MatrixCalculator


class NeuralNetwork:
    def __init__(self):
        self.layers = {}
        self.weights = {}
        self.calc = MatrixCalculator()

        self.max_allowable_error = 0.7
        self.total_error = 0
        self.quantity_of_iterarions = 0

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

        def __str__ (self):
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

            self.learning_rate = random() / 100

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
            self.MatrixOfWeights(matrix_of_weights_id, layer_id_first, layer_id_second, self).createMatrix(list_of_weights)

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

    def trainWeightsLast(self, matrix_of_weights_id: str, delta_layer_ids: tuple, previous_layer_id: int):
        # W'(i+1) = W'(i) – α2*[Y(i)]T * dX(i)
        old_weights = copy(self.weights[matrix_of_weights_id])  # W'(i)
        delta_matrix = self.calc.Matrix()
        delta_matrix.create(elements=self.getDeltaLayers(layer_id_first=delta_layer_ids[0],
                                                         layer_id_second=delta_layer_ids[1]),
                            width=self.layers[delta_layer_ids[0]].count_of_neurons,
                            height=1)  # dX(i)
        previous_layer = self.layers[previous_layer_id]
        previous_layer_matrix = self.calc.Matrix()
        previous_layer_matrix.create(elements=[neuron.value for neuron in previous_layer.neurons],
                                     width=previous_layer.count_of_neurons, height=1)
        deductible_p1 = self.calc.multiple(self.calc.transpose(previous_layer_matrix), delta_matrix)  # [Y(i)]T * dX(i)
        deductible = self.calc.multipleNum(matrix=deductible_p1,
                                           num=self.weights[matrix_of_weights_id].learning_rate)  # α2 * [Y(i)]T * dX(i) W'(i)->a'
        new_weights = self.calc.diff(old_weights, deductible)  # W'(i) – α2 * [Y(i)]T * dX(i)
        self.weights[matrix_of_weights_id].matrix = new_weights.matrix
        # self.weights[matrix_of_weights_id].matrix = self.calc.diff(self.weights[matrix_of_weights_id], deductible).matrix
        # print("NEW WEIGHTS:\n", self.weights[matrix_of_weights_id].matrix, "\n")

    def trainWeights(self, matrix_of_weights_first_id: str, matrix_of_weights_second_id: str, delta_layer_ids: tuple, previous_layer_id: int):
        # W(i+1) = W(i) – α * [X(i)]T * dX(i) * [W'(i)]T
        old_weights_first = copy(self.weights[matrix_of_weights_first_id])  # W(i)
        old_weights_second_transposed = self.calc.transpose(copy(self.weights[matrix_of_weights_second_id]))  # [W'(i)]T
        delta_matrix = self.calc.Matrix()
        delta_matrix.create(elements=self.getDeltaLayers(layer_id_first=delta_layer_ids[0],
                                                         layer_id_second=delta_layer_ids[1]),
                            width=self.layers[delta_layer_ids[0]].count_of_neurons,
                            height=1)  # dX(i)
        previous_layer = self.layers[previous_layer_id]
        previous_layer_matrix = self.calc.Matrix()
        previous_layer_matrix.create(elements=[neuron.value for neuron in previous_layer.neurons],
                                     width=previous_layer.count_of_neurons, height=1)  # X(i)

        deductible_p1 = self.calc.multiple(self.calc.transpose(previous_layer_matrix), delta_matrix)  # [X(i)]T * dX(i)
        deductible_p2 = self.calc.multiple(deductible_p1, old_weights_second_transposed)  # [X(i)]T * dX(i) * [W'(i)]T
        deductible = self.calc.multipleNum(matrix=deductible_p2,
                                           num=self.weights[matrix_of_weights_first_id].learning_rate)  # α * [X(i)]T * dX(i) * [W'(i)]T  W(i)->a
        new_weights = self.calc.diff(old_weights_first, deductible)  # W(i) – α * [X(i)]T * dX(i) * [W'(i)]T
        self.weights[matrix_of_weights_first_id].matrix = new_weights.matrix

    def normalizeWeights(self, matrix_of_weights_id: str):
        transposed_matrix = self.calc.transpose(self.weights[matrix_of_weights_id])

        for i in range(0, self.weights[matrix_of_weights_id].height):
            for j in range(0, self.weights[matrix_of_weights_id].width):
                # module = 1
                module = 1
                # for k in transposed_matrix.matrix[j]:
                #     module *= k
                for k in transposed_matrix.matrix[j]:
                    module *= k
                self.weights[matrix_of_weights_id].matrix[i][j] *= (module / len(transposed_matrix.matrix[j]))

    def calculateTotalRMSError(self, delta_layer_ids: tuple):
        # Е(q) = E(dX(q)i * dX(q)i), where 1<=i<=N
        delta = self.getDeltaLayers(layer_id_first=delta_layer_ids[0], layer_id_second=delta_layer_ids[1])
        RMS = 0
        for value in delta:
            RMS += value * value
        return RMS

    ##### GLOBAL NETWORK FUNCTIONS

    def trainNeuronNetwork(self, blocks):
        self.creatingNetwork(blocks=blocks)
        # for _ in range(0, 3):
        print(self.weights["2-3"])

        while self.total_error > self.max_allowable_error or self.total_error == 0:
            for block in blocks:
                self.trainingIteration(block)
                print(self.weights["2-3"])
                #print("ITERATION: ", self.quantity_of_iterarions, "\t errors sum: ", self.total_error,
                #      "\t max_error: ", self.max_allowable_error, "\t current average error: ", self.total_error/len(blocks))
            # if self.total_error/len(blocks) < self.max_allowable_error:
            if self.total_error/len(blocks) < self.max_allowable_error or self.quantity_of_iterarions > 500:
                break
            self.total_error = 0

    def creatingNetwork(self, blocks):
        # create layers
        block_length = len(blocks[0])
        self.createLayer(layer_id=1, count_of_neurons=block_length)
        self.createLayer(layer_id=2, count_of_neurons=int(block_length * 2))
        self.createLayer(layer_id=3, count_of_neurons=block_length)
        # crete matrices of weights 1-2/2-3
        self.createWeightsRandom(matrix_of_weights_id="1-2", layer_id_first=1, layer_id_second=2)
        weights = self.calc.getListFromMatrix(self.calc.transpose(self.weights["1-2"]))
        self.createWeights(matrix_of_weights_id="2-3", layer_id_first=2, layer_id_second=3, list_of_weights=weights)

        self.max_allowable_error *= block_length

    def trainingIteration(self, block):
        self.fillLayer(layer_id=1, list_of_values=block)

        self.trainNeurons(layer_id_first=1, layer_id_second=2, matrix_of_weights_id="1-2")  # training 2 layer
        self.trainNeurons(layer_id_first=2, layer_id_second=3, matrix_of_weights_id="2-3")  # training 3 layer

        self.trainWeights(matrix_of_weights_first_id="1-2", matrix_of_weights_second_id="2-3", delta_layer_ids=(1, 3), previous_layer_id=1)
        self.trainWeightsLast(matrix_of_weights_id="2-3", delta_layer_ids=(1, 3), previous_layer_id=2)

        self.normalizeWeights("1-2")
        self.normalizeWeights("2-3")

        self.total_error += self.calculateTotalRMSError(delta_layer_ids=(1, 3))
        self.quantity_of_iterarions += 1

    def runBlockThroughNN(self, block):
        self.fillLayer(layer_id=1, list_of_values=block)
        self.trainNeurons(layer_id_first=1, layer_id_second=2, matrix_of_weights_id="1-2")  # training 2 layer
        self.trainNeurons(layer_id_first=2, layer_id_second=3, matrix_of_weights_id="2-3")  # training 3 layer
        return [neuron.value for neuron in self.layers[3].neurons]