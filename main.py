from ImageCompressorClass import ImageCompressor

block_sizes = (1, 1)
image_file = "images/1.jpeg"

compressed_image = ImageCompressor()
compressed_image.compressImage(image_file, block_sizes)
