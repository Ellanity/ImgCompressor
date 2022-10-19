# from ImageConverterClass import ImageConverter
from ImageCompressorClass import ImageCompressor
from MatrixCalculatorClass import MatrixCalculator

"""
block_sizes = (2, 2)
image_file = "5x5.jpg"

compressed_image = ImageCompressor()
compressed_image.compressImage(image_file, block_sizes)

print(compressed_image.blocks)

"""
calc = MatrixCalculator()
matrix_1 = calc.Matrix().create([9, 3, 5, 2, 0, 3, 0, 1, -6], 3, 3)
matrix_2 = calc.Matrix().create([1, -1, -1, -1, 4, 7, 8, 1, -1], 3, 3)
matrix_3 = calc.Matrix().create([1, 2, 3, 4, 5, 6], 2, 3)
matrix_1.print()
print("")
matrix_2.print()
print("")
calc.sum(matrix_1, matrix_2).print()
print("")
calc.diff(matrix_1, matrix_2).print()
print("")
calc.multiple(matrix_1, matrix_2).print()
print("")
calc.transpose(matrix_3).print()
print("")
calc.multipleNum(matrix_3, 3.2).print()
print("")

