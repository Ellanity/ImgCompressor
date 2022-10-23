import datetime
import random
import math

import matplotlib.image as img
import PIL.Image as PilImg

from NeuralNetworkClass import NeuralNetwork


class ImageCompressor:

    def __init__(self):
        self.image_file_name = ""
        # convert
        self.pixel_size = 0
        self.image_size = [0, 0]
        self.image_array = []
        self.image_vector = []
        self.image_vector_converted = []
        # compress
        self.block_width = 0
        self.block_height = 0
        self.blocks = []
        self.nn = NeuralNetwork()
        self.compressed_blocks = []
        # restore
        self.vector_of_compressed_values = []
        self.image_vector_restored = []
        self.image_array_restored = []

    def compressImage(self, image, block_sizes):
        # init
        print("set image")
        self.__setImageFile(image)
        self.__setBlocksSizes(block_sizes)
        # convert img to vector of values
        print("convert image")
        self.__getImageArray()
        self.__straightenImageArray()
        self.__convertValuesInImageArray()
        # convert vector in vector of blocks/vectors
        print("split image")
        self.__splitImageArrayIntoBlocks()
        # work with nn
        print("create and train nn")
        self.__createNN()
        print("compress image")
        self.__compressImageWithNN()
        # self.compressed_blocks = self.blocks
        # restore
        print("restore image")
        self.__restoreArrayFromBlocks()
        self.__restoreValuesInImageArray()
        self.__collapseRestoredArray()
        self.__saveImage()

    def __setImageFile(self, image_file_name):
        self.image_file_name = image_file_name

    # convert
    def __setBlocksSizes(self, block_sizes):
        self.block_width = block_sizes[0]
        self.block_height = block_sizes[1]

    def __getImageArray(self):
        file = PilImg.open(self.image_file_name, mode='r')
        self.image_array = img.pil_to_array(file)
        self.image_mode = file.mode

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
                            index = (first_row_of_block + number_of_row_in_block) * \
                                    self.image_size[0] * self.pixel_size + \
                                    (first_column_of_block * self.pixel_size + number_of_column_in_block)
                            # print(index)
                            new_block.append(self.image_vector_converted[index])
                        except Exception as _:
                            pass  # print(ex)
                # print("end block")
                self.blocks.append(new_block)

    # compress
    def __createNN(self):
        rand_blocks = [random.choice(list(self.blocks)) for _ in range(0, 1000)]
        self.nn.trainNeuronNetwork(list(rand_blocks))

    def __compressImageWithNN(self):
        self.compressed_blocks = []
        for block in self.blocks:
            self.compressed_blocks.append(self.nn.runBlockThroughNN(block))

    # restore
    def __restoreArrayFromBlocks(self):
        # print(self.compressed_blocks)
        self.vector_of_compressed_values = [0] * (self.image_size[0] * self.image_size[1] * self.pixel_size)
        count_of_blocks_in_column = math.ceil(self.image_size[1] / self.block_height)
        count_of_blocks_in_row = math.ceil(self.image_size[0] / self.block_width)
        for number_of_block_in_column in range(0, count_of_blocks_in_column):
            for number_of_block_in_row in range(0, count_of_blocks_in_row):
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
                            i = (first_row_of_block + number_of_row_in_block) * self.image_size[0] * self.pixel_size
                            j = (first_column_of_block * self.pixel_size + number_of_column_in_block)
                            bi = number_of_block_in_column * count_of_blocks_in_row + number_of_block_in_row
                            bj = number_of_row_in_block * self.block_width * self.pixel_size + number_of_column_in_block
                            self.vector_of_compressed_values[i + j] = self.compressed_blocks[bi][bj]
                        except Exception as _:
                            pass  # print(ex)

    def __restoreValuesInImageArray(self):
        self.image_vector_restored = []
        for value in self.vector_of_compressed_values:
            new_value = 255 * ((value + 1) / 2)
            self.image_vector_restored.append(new_value)

    def __collapseRestoredArray(self):
        # self.image_array_restored = \
        # np.array([np.array([np.array([0] * self.pixel_size)]*self.image_size[0])]*self.image_size[1])
        self.image_array_restored = self.image_array.copy()
        value_index = 0
        row_index = 0
        pixel_index = 0
        for value in self.image_vector_restored:
            self.image_array_restored[row_index][pixel_index][value_index % self.pixel_size] = value  # 255 - value
            # print(value / 128)
            value_index += 1
            if value_index % self.pixel_size == 0:
                pixel_index += 1
            if pixel_index % self.image_size[0] == 0 and pixel_index != 0:
                row_index += 1
                pixel_index = 0

    def __saveImage(self):
        new_ing = PilImg.fromarray(self.image_array_restored, self.image_mode)
        new_ing.show()
        new_ing.save(f'{self.image_file_name}-{datetime.datetime.now()}.png'.replace(":", "_").replace(" ", "_"))
