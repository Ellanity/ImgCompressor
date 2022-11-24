import random
from ImageClass import Image
from NNClass import NN


def main():
    random.seed()
    # User input
    image_file_name = input("Image file name: ")
    image_file_name = "images/256x256_0.png" \
        if image_file_name == "" or image_file_name == "\n" else image_file_name

    block_width = input("Block width: ")
    block_width = 1 \
        if block_width == "" or block_width == "\n" else int(block_width)

    block_height = input("Block height: ")
    block_height = 1 \
        if block_height == "" or block_height == "\n" else int(block_height)

    coeff_quantity_neurons_second_layer = input("Coeff of quantity of neurons on the second layer: ")
    coeff_quantity_neurons_second_layer = 2 \
        if coeff_quantity_neurons_second_layer == "" \
           or coeff_quantity_neurons_second_layer == "\n" else float(coeff_quantity_neurons_second_layer)

    learning_rate = input("Learning rate: ")
    learning_rate = 0.0001 \
        if learning_rate == "" or learning_rate == "\n" else float(learning_rate)

    can_load_weights = input("Load saved weights [y/n]: ")
    can_load_weights = True \
        if can_load_weights.lower() == "y" or can_load_weights.lower() == "y\n" else False

    # !!!Only for dev always True/False
    can_load_weights = True

    NNVariables = {
        "image_name": image_file_name,
        "block_width": block_width,
        "block_height": block_height,
        "coeff_quantity_neurons_second_layer": coeff_quantity_neurons_second_layer,
        "quantity_of_blocks_to_train": 1000,
        "learning_rate": learning_rate,
        "can_load_weights": can_load_weights
    }
    print(NNVariables)
    # Splitting an image into channels and blocks
    image = Image(NNVariables["image_name"])
    image.splitImageChannelsIntoBlocks(NNVariables["block_width"], NNVariables["block_height"])

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
        # channel_NN.loadWeights()
        channel_NN.trainNN([random.choice(channel.blocks)
                            for _ in range(0, NNVariables["quantity_of_blocks_to_train"])])
        # Run all channel blocks through a neural network
        iterated_blocks = []
        for block in channel.blocks:
            iterated_blocks.append(channel_NN.NNIteration(block))
        channel.blocks = iterated_blocks
        channel_NN.saveWeights()

    # Restore and save imag
    image.restoreChannelsFromBlocks(NNVariables["block_width"], NNVariables["block_height"])
    image.restoreImageFromChannels()
    image.saveImage()

    # Literature:
    # https://studfile.net/preview/1557061/page:8/


if __name__ == '__main__':
    main()
