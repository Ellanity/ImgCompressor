import random
from ImageClass import Image
from NNClass import NN


def main():
    random.seed()
    NNVariables = {
        "image_name": "images/256x256_0.png",
        "block_width": 1,
        "block_height": 1,
        "coeff_of_quantity_of_neurons_second_layer": 2,
        "quantity_of_blocks_to_train": 1000,
        "learning_rate": 0.0001
    }

    image = Image(NNVariables["image_name"])
    image.splitImageChannelsIntoBlocks(NNVariables["block_width"], NNVariables["block_height"])

    for channel in image.channels:
        # if image.channels.index(channel) != 0:
        #     break
        channel_NN = NN()
        NNVariables["channel_id"] = image.channels.index(channel)
        channel_NN.createNN(NNVariables)
        channel_NN.trainNN([random.choice(channel.blocks) for _ in range(0, NNVariables["quantity_of_blocks_to_train"])])

        iterated_blocks = []
        for block in channel.blocks:
            iterated_blocks.append(channel_NN.NNIteration(block))
        channel.blocks = iterated_blocks

    image.restoreChannelsFromBlocks(NNVariables["block_width"], NNVariables["block_height"])
    image.restoreImageFromChannels()
    image.saveImage()


if __name__ == '__main__':
    main()
