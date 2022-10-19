"""
import matplotlib.image as img
import PIL.Image as PilImg
import math


class ImageConverter:

    def __init__(self, image_file_name, block_sizes):
        self.image_file_name = image_file_name
        self.image_array = []

        self.block_width = block_sizes[0]
        self.block_height = block_sizes[1]
        self.blocks = []

    def setImageFile(self, image_file_name):
        self.image_file_name = image_file_name

    def getArrayFromImage(self):
        file = PilImg.open(self.image_file_name, mode='r')
        self.image_array = img.pil_to_array(file)

    def setBlocksSizes(self, block_sizes):
        self.block_width = block_sizes[0]
        self.block_height = block_sizes[1]

    def splitArrayIntoBlocks(self):
        if len(self.image_array) == 0:
            raise Exception("No array from image")

        self.blocks.clear()
        image_width = len(self.image_array[0])
        image_height = len(self.image_array)
        count_blocks_in_row = image_width / self.block_width
        count_blocks_in_column = image_height / self.block_height

        for num_of_bloc_in_column in range(0, math.ceil(count_blocks_in_column)):
            first_row = num_of_bloc_in_column * self.block_height
            last_row = (num_of_bloc_in_column + 1) * self.block_height

            for num_of_bloc_in_row in range(0, math.ceil(count_blocks_in_row)):
                first_pix_in_row = num_of_bloc_in_row * self.block_width
                last_pix_in_row = (num_of_bloc_in_row + 1) * self.block_width
                new_block = []

                for row in range(first_row, last_row):
                    row_in_block = []

                    for pix in range(first_pix_in_row, last_pix_in_row):
                        try:
                            row_in_block.append(list(self.image_array[row][pix]))
                        except Exception as ex:
                            pass  # print(ex)
                    if len(row_in_block) != 0:
                        new_block.append(row_in_block)
                self.blocks.append(new_block)
"""