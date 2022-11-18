import datetime
import math
import matplotlib.image
import PIL.Image as PilImg
from pathlib import Path

from MatrixClass import Matrix


class Image:

    def __init__(self, file_name: str):
        self.file_name = file_name
        self.width = 0
        self.height = 0
        self.pixels_array = []
        self.mode = None
        self.channels = []
        self.__loadImage()

    def __loadImage(self):
        with PilImg.open(self.file_name, mode='r') as file:
            self.pixels_array = matplotlib.image.pil_to_array(file)
            self.mode = file.mode
            self.number_of_channels = len(self.pixels_array[0][0])
            self.width = len(self.pixels_array[0])
            self.height = len(self.pixels_array)
            self.__splitImageIntoChannels()

    def __splitImageIntoChannels(self):
        self.channels.clear()
        # create channels
        for channel_index in range(0, self.number_of_channels):
            self.channels.append(self.Channel(width=self.width, height=self.height,
                                              elements=[0 for _ in range(self.width * self.height)]))

        # fill channels
        for row_index in range(0, self.height):
            for pixel_index in range(0, self.width):
                for channel_index in range(0, self.number_of_channels):
                    self.channels[channel_index].matrix[row_index][pixel_index] = \
                        self.pixels_array[row_index][pixel_index][channel_index]

        # refresh elements in matrices objects
        # for channel in self.channels:
        #     channel.getList()

    class Channel(Matrix):
        def __init__(self, width: int, height: int, elements: list):
            super().__init__(width=width, height=height, elements=elements)
            self.blocks = []

        def splitChannelIntoBlocks(self, block_width: int, block_height: int):
            self.blocks.clear()
            count_of_blocks_in_column = math.ceil(self.height / block_height)
            count_of_blocks_in_row = math.ceil(self.width / block_width)

            for number_of_block_in_column in range(0, count_of_blocks_in_column):
                for number_of_block_in_row in range(0, count_of_blocks_in_row):
                    new_block = []
                    # alternative: block from matrix
                    # new_block = Matrix(width=block_width, height=block_height,
                    #                    elements=[0 for _ in range(block_width * block_height)])
                    # get the indexes of the first element of the block relative to the entire image
                    first_column_of_block = number_of_block_in_row * block_width
                    first_row_of_block = number_of_block_in_column * block_height

                    # move last block in row
                    if number_of_block_in_row == count_of_blocks_in_row - 1:
                        rest_of_pixels = self.width - block_width * (count_of_blocks_in_row - 1)
                        if rest_of_pixels > 0:
                            first_column_of_block -= (block_width - rest_of_pixels)

                    # move last block in column
                    if number_of_block_in_column == count_of_blocks_in_column - 1:
                        rest_of_pixels = self.height - block_height * (count_of_blocks_in_column - 1)
                        if rest_of_pixels > 0:
                            first_row_of_block -= (block_height - rest_of_pixels)

                    for number_of_row_in_block in range(0, block_height):
                        for number_of_column_in_block in range(0, block_width):
                            try:
                                element = self.matrix[first_row_of_block + number_of_row_in_block][first_column_of_block + number_of_column_in_block]
                                element = (2 * element / 255) - 1  # for tests !must comment it!
                                element = round(element, 1)
                                new_block.append(element)
                                # alternative: block from matrix
                                # new_block.matrix[number_of_row_in_block][number_of_column_in_block] = element
                            except Exception as ex:
                                print(ex)
                    self.blocks.append(new_block)

        def restoreChannelFromBlocks(self, block_width: int, block_height: int):
            count_of_blocks_in_column = math.ceil(self.height / block_height)
            count_of_blocks_in_row = math.ceil(self.width / block_width)

            for number_of_block_in_column in range(0, count_of_blocks_in_column):
                for number_of_block_in_row in range(0, count_of_blocks_in_row):
                    first_column_of_block = number_of_block_in_row * block_width
                    first_row_of_block = number_of_block_in_column * block_height

                    # move last block in row
                    if number_of_block_in_row == count_of_blocks_in_row - 1:
                        rest_of_pixels = self.width - block_width * (count_of_blocks_in_row - 1)
                        if rest_of_pixels > 0:
                            first_column_of_block -= (block_width - rest_of_pixels)

                    # move last block in column
                    if number_of_block_in_column == count_of_blocks_in_column - 1:
                        rest_of_pixels = self.height - block_height * (count_of_blocks_in_column - 1)
                        if rest_of_pixels > 0:
                            first_row_of_block -= (block_height - rest_of_pixels)

                    for number_of_row_in_block in range(0, block_height):
                        for number_of_column_in_block in range(0, block_width):
                            try:
                                block_num = number_of_block_in_column * count_of_blocks_in_row + number_of_block_in_row
                                element_in_block_num = number_of_row_in_block * block_width + number_of_column_in_block
                                channel_i = first_row_of_block + number_of_row_in_block
                                channel_j = first_column_of_block + number_of_column_in_block
                                element = self.blocks[block_num][element_in_block_num]
                                # print(self.blocks[block_num][element_in_block_num])
                                element = 255 * ((element + 1) / 2)
                                # self.blocks[block_num][element_in_block_num] = element
                                # print(self.blocks[block_num][element_in_block_num])
                                self.matrix[channel_i][channel_j] = element
                            except Exception as ex:
                                print(ex)

    def splitImageChannelsIntoBlocks(self, block_width: int, block_height: int):
        for channel in self.channels:
            channel.splitChannelIntoBlocks(block_width=block_width, block_height=block_height)
        # print(self.channels[0])

    def restoreChannelsFromBlocks(self, block_width: int, block_height: int):
        #for channel in self.channels:
        #    print(channel.blocks)
        for channel in self.channels:
            channel.restoreChannelFromBlocks(block_width=block_width, block_height=block_height)
        #for channel in self.channels:
        #    print(channel.blocks)

    def restoreImageFromChannels(self):
        # fill channels
        for row_index in range(0, self.height):
            for pixel_index in range(0, self.width):
                for channel_index in range(0, self.number_of_channels):
                    self.pixels_array[row_index][pixel_index][channel_index] = \
                        self.channels[channel_index].matrix[row_index][pixel_index]

    def saveImage(self):

        new_ing = PilImg.fromarray(self.pixels_array, self.mode)
        new_ing.show()
        Path(f"images/{Path(self.file_name).stem}").mkdir(parents=True, exist_ok=True)
        new_ing.save(f'images/{Path(self.file_name).stem}/{datetime.datetime.now()}.png'.replace(":", "_").replace(" ", "_"))