import random
import math

import matplotlib.image as Img
import PIL.Image as PilImg

from NeuralNetworkClass import NeuralNetwork


class ImageCompressor:

    def __init__(self):
        self.image_file_name = ""
        self.image_array = []
        self.image_vector = []
        self.image_vector_converted = []

        self.pixel_size = 0
        self.image_size = [0, 0]

        self.block_width = 0
        self.block_height = 0
        self.blocks = []

        self.nn = NeuralNetwork()
        self.compressed_blocks = []

        self.matrix_of_weights_first = []
        self.matrix_of_weights_second = []
        self.count_of_neurons_on_second_layer = 0

    def compressImage(self, image, block_sizes):
        # init
        self.__setImageFile(image)
        self.__setBlocksSizes(block_sizes)
        # convert Img to vector of values
        self.__getImageArray()
        self.__straightenImageArray()
        self.__convertValuesInImageArray()
        # convert vector in vector of blocks/vectors
        self.__splitImageArrayIntoBlocks()
        # work with nn
        self.__createNN()
        self.__compressImageWithNN()


    def __setImageFile(self, image_file_name):
        self.image_file_name = image_file_name

    def __setBlocksSizes(self, block_sizes):
        self.block_width = block_sizes[0]
        self.block_height = block_sizes[1]

    def __getImageArray(self):
        file = PilImg.open(self.image_file_name, mode='r')
        self.image_array = Img.pil_to_array(file)

    def __straightenImageArray(self):
        self.image_vector.clear()
        self.pixel_size = 0
        self.image_size = [0, 0]

        # remember pixel size
        if self.pixel_size == 0:
            self.pixel_size = len(self.image_array[0][0])

        # remember picture size
        if self.image_size == [0, 0]:
            self.image_size[0] = len(self.image_array[0])
            self.image_size[1] = len(self.image_array)

        for row in self.image_array:
            for pixel in row:
                for channel in pixel:
                    self.image_vector.append(channel)

    def __convertValuesInImageArray(self):
        for value in self.image_vector:
            new_value = (2 * value / 255) - 1
            self.image_vector_converted.append(new_value)

    def __splitImageArrayIntoBlocks(self):
        self.blocks.clear()
        count_of_blocks_in_column = math.ceil(self.image_size[1] / self.block_height)
        count_of_blocks_in_row = math.ceil(self.image_size[0] / self.block_width)

        for number_of_block_in_column in range(0, count_of_blocks_in_column):
            for number_of_block_in_row in range(0, count_of_blocks_in_row):
                new_block = []
                first_column_of_block = number_of_block_in_row * self.block_width
                first_row_of_block = number_of_block_in_column * self.block_height

                if number_of_block_in_row == count_of_blocks_in_row - 1:
                    rest_of_pixels = self.image_size[0] - self.block_width * (count_of_blocks_in_row - 1)
                    if rest_of_pixels > 0:
                        first_column_of_block -= (self.block_width - rest_of_pixels)

                if number_of_block_in_column == count_of_blocks_in_column - 1:
                    rest_of_pixels = self.image_size[1] - self.block_height * (count_of_blocks_in_column - 1)
                    if rest_of_pixels > 0:
                        first_row_of_block -= (self.block_height - rest_of_pixels)

                for number_of_row_in_block in range(0, self.block_height):
                    for number_of_column_in_block in range(0, self.block_width * self.pixel_size):
                        try:
                            index = (first_row_of_block + number_of_row_in_block) * self.image_size[0] * self.pixel_size + \
                                    (first_column_of_block * self.pixel_size + number_of_column_in_block)
                            # print(index)
                            new_block.append(self.image_vector_converted[index])
                        except Exception as ex:
                            pass  # print(ex)
                # print("end block")
                self.blocks.append(new_block)
        # count of neurons on second layer
        self.count_of_neurons_on_second_layer = len(self.blocks[0]) * 2

    def __createNN(self):
        rand_blocks = [random.choice(list(self.blocks)) for _ in range(0, 10)]
        self.nn.trainNeuronNetwork(list(rand_blocks))

    def __compressImageWithNN(self):
        self.compressed_blocks = []
        for block in self.blocks:
            self.compressed_blocks.append(self.nn.runBlockThroughNN(block))
