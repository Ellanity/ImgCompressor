import random

from ImageCompressorClass import ImageCompressor
# from MatrixCalculatorClass import MatrixCalculator
# from NeuralNetworkClass import NeuralNetwork

block_sizes = (4, 4)
image_file = "images/1.jpeg"

compressed_image = ImageCompressor()
compressed_image.compressImage(image_file, block_sizes)
# print(compressed_image.compressed_blocks)

"""rand_blocks = [random.choice(list(compressed_image.blocks)) for _ in range(0, 10)]
nn = NeuralNetwork()
nn.trainNeuronNetwork(list(rand_blocks))"""
