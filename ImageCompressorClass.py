import random
from ImageClass import Image
from NeuralNetworkClass import NeuralNetwork


class ImageCompressor:

    def __init__(self):
        self.image = None
        self.image_name = None
        self.block_width = 0
        self.block_height = 0
        self.quantity_of_blocks_to_train_neural_network = 0

    def setImage(self, image_name):
        self.image_name = image_name
        self.image = Image(image_name)

    def setBlockSize(self, block_width, block_height):
        self.block_width = block_width
        self.block_height = block_height

    def setQuantityOfBlocksToTrainNeuralNetwork(self, quantity_of_blocks_to_train_neural_network):
        self.quantity_of_blocks_to_train_neural_network = quantity_of_blocks_to_train_neural_network

    def compressImage(self):
        if self.image is None:
            raise Exception("No image to create neural network")
        self.image.splitImageChannelsIntoBlocks(block_width=self.block_width, block_height=self.block_height)

        #for channel in self.image.channels:
        #    print(channel.blocks)
        for channel in self.image.channels:
            channel.blocks = self.compressImageChannel(channel)
        #for channel in self.image.channels:
        #    print(channel.blocks)
        self.image.restoreChannelsFromBlocks(block_width=self.block_width, block_height=self.block_height)
        self.image.restoreImageFromChannels()

        self.image.saveImage()

    def compressImageChannel(self, channel):
        neural_network = self.createNeuralNetworkForBlocks(channel.blocks)
        return self.runAllBlocksThroughNeuralNetwork(neural_network, channel.blocks)

    def createNeuralNetworkForBlocks(self, blocks):
        random_blocks = [random.choice(blocks) for _ in range(0, self.quantity_of_blocks_to_train_neural_network)]
        neural_network = NeuralNetwork()
        neural_network.trainNeuronNetwork(random_blocks)
        return neural_network

    def runAllBlocksThroughNeuralNetwork(self, neural_network, blocks):
        compressed_blocks = []
        # print("blocks", blocks)
        for block in blocks:
            compressed_blocks.append(neural_network.runBlockThroughNeuralNetwork(block))
        # print("compressed_blocks", compressed_blocks)
        return compressed_blocks
