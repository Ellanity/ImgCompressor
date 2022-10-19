from ImageCompressorClass import ImageCompressor
from MatrixCalculatorClass import MatrixCalculator

block_sizes = (2, 2)
image_file = "images/5x5.jpg"

compressed_image = ImageCompressor()
compressed_image.compressImage(image_file, block_sizes)

print(compressed_image.blocks)
