from ImageCompressorClass import ImageCompressor

block_sizes = (1, 1)
image_file = "images/256x256.jpg"
quantity_of_blocks_for_nn = 1024

compressed_image = ImageCompressor()
compressed_image.compressImage(image_file, block_sizes, quantity_of_blocks_for_nn)
