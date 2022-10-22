from ImageCompressorClass import ImageCompressor

block_sizes = (2, 2)
image_file = "images/4.png"

compressed_image = ImageCompressor()
compressed_image.compressImage(image_file, block_sizes)
