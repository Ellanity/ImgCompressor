import random
from ImageClass import Image
from NNClass import NN, Matrix


def uncompressImage():
    # User input
    image_file_name = input("Image file name: ")
    image_file_name = "images/5x5/compressed/5_5_0.7_3_3_2022-12-10_02_34_58.875684.png" \
        if image_file_name == "" or image_file_name == "\n" else image_file_name
    image = Image(image_file_name)
    image.splitImageChannelsIntoBlocks(block_width=1, block_height=image.height)
    splited_path = image_file_name.split("/")
    splited_image_name = None
    try:
        splited_image_name = splited_path[3].split("_")
    except Exception as _:
        splited_image_name = splited_path[2].split("_")

    if splited_image_name is None:
        print("File name format is incorrect")
        return

    image.width = int(splited_image_name[0])
    image.height = int(splited_image_name[1])
    NNVariables = {
        "image_name": f"images/{splited_path[1]}",
        "block_width": int(splited_image_name[3]),
        "block_height": int(splited_image_name[4]),
        "compression_rate": float(splited_image_name[2]),
        "quantity_of_blocks_to_train": 1000,
        "learning_rate": 0,
        "can_load_weights": True
    }
    image.setNNVariables(NNVariables)
    # print("uncompressed")
    # _ = [print(channel.blocks) for channel in image.channels]

    for channel in image.channels:
        channel_NN = NN()
        NNVariables["channel_id"] = image.channels.index(channel)
        NNVariables["count_of_channels"] = len(image.channels)
        channel_NN.createNN(NNVariables)
        blocks_output = []
        for block in channel.blocks:
            layer_2 = Matrix(width=len(block), height=1, elements=block)
            blocks_output.append((layer_2 * channel_NN.weights_2).getList())
        channel.blocks = blocks_output
        channel.width = int(splited_image_name[0])
        channel.height = int(splited_image_name[1])
        channel.matrix = [[0] * channel.width for _ in range(channel.height)]
        # channel.restoreChannelFromBlocks(block_width=NNVariables["block_width"],
        #                                  block_height=NNVariables["block_height"])
        # print(*channel.blocks)
    image.restoreChannelsFromBlocks(block_width=NNVariables["block_width"],
                                    block_height=NNVariables["block_height"])
    image.restoreImageFromChannels()
    image.saveImage()


def compressImage():
    random.seed()
    # User input
    image_file_name = input("Image file name: ")
    # image_file_name = "images/256x256_0.png" if image_file_name == "" or image_file_name == "\n" else image_file_name
    image_file_name = "images/5x5.jpg" if image_file_name == "" or image_file_name == "\n" else image_file_name

    block_width = input("Block width: ")
    block_width = 3 if block_width == "" or block_width == "\n" else int(block_width)

    block_height = input("Block height: ")
    block_height = 3 if block_height == "" or block_height == "\n" else int(block_height)

    compression_rate = input("Coeff of quantity of neurons on the second layer: ")
    compression_rate = 0.7 if compression_rate == "" or compression_rate == "\n" else float(compression_rate)

    learning_rate = input("Learning rate: ")
    learning_rate = 0.0001 if learning_rate == "" or learning_rate == "\n" else float(learning_rate)

    # can_load_weights = input("Load saved weights [y/n]: ")
    # can_load_weights = True if can_load_weights.lower() == "y" or can_load_weights.lower() == "y\n" else False

    # !!!Only for dev always True/False
    can_load_weights = True

    NNVariables = {
        "image_name": image_file_name,
        "block_width": block_width,
        "block_height": block_height,
        "compression_rate": compression_rate,
        "quantity_of_blocks_to_train": 1000,
        "learning_rate": learning_rate,
        "can_load_weights": can_load_weights
    }
    print(NNVariables)
    # Splitting an image into channels and blocks
    image = Image(NNVariables["image_name"])
    image.setNNVariables(NNVariables)
    image.splitImageChannelsIntoBlocks(NNVariables["block_width"], NNVariables["block_height"])
    NNVariables["image_width"] = image.width
    NNVariables["image_height"] = image.height

    """ 
    It's possible to create only one neural network instead of three(four),
    for this u just need uniform sampling from all channels of the image
    """

    """
    samples = [] 
    for channel in image.channels:
        samples += [random.choice(channel.blocks) 
            for _ in range(0, int(NNVariables["quantity_of_blocks_to_train"] / 3))]

    NNVariables["channel_id"] = 0
    common_NN = NN()
    common_NN.createNN(NNVariables)
    common_NN.trainNN(samples)
    
    for channel in image.channels:
        iterated_blocks = []
        for block in channel.blocks:
            iterated_blocks.append(common_NN.NNIteration(block))
        channel.blocks = iterated_blocks
    """

    """
    Three(four) neural networks, one for every channel   
    with the same error, the quality of the resulting image
    when using a neural network for each image channel is better
    than when using a single neural network
    """

    for channel in image.channels:
        # Create and train neural network
        channel_NN = NN()
        NNVariables["channel_id"] = image.channels.index(channel)
        NNVariables["count_of_channels"] = len(image.channels)
        channel_NN.createNN(NNVariables)
        channel_NN.trainNN([random.choice(channel.blocks)
                            for _ in range(0, NNVariables["quantity_of_blocks_to_train"])])
        # Run all channel blocks through a neural network
        iterated_blocks = []
        for block in channel.blocks:
            iterated_blocks.append(channel_NN.NNIteration(block))
        channel.blocks = [block["final"] for block in iterated_blocks]
        image.channels_compressed[image.channels.index(channel)] = [block["compressed"] for block in iterated_blocks]

    # print("compressed")
    # _ = [print(channel) for channel in image.channels_compressed]
    # Restore and save image and save compressed image
    image.restoreChannelsFromBlocks(NNVariables["block_width"], NNVariables["block_height"])
    image.restoreImageFromChannels()
    image.saveImage()
    image.saveImageCompressedVersion()
    # print(image.pixels_array)

    # Literature:
    # https://studfile.net/preview/1557061/page:8/


def main():
    """
    main_program_flow = 0
    while main_program_flow != "1" and main_program_flow != "2":
        # main_program_flow = input("What do you want to do:\n1 - compress image\n2 - uncompress image\n")
        main_program_flow = "1"
    if main_program_flow == "1":
        compressImage()
    elif main_program_flow == "2":
        uncompressImage()
    else:
        print("Exit program")
        exit()
    """
    compressImage()
    # uncompressImage()


if __name__ == '__main__':
    main()
