from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .simplecnn import CNN
from .simplecnn2 import Net_circular_CNN
from .vgg import VGG
from .wideresnet import ResNet16_1, ResNet16_2, ResNet16_3, ResNet16_4, ResNet16_5, ResNet16_6, ResNet16_10, ResNet28_10

# Resnet
def resnet18(num_classes = 10, norm_layer_type = 'bn' ,conv_layer_type = 'conv2d',linear_layer_type = 'linear', activation_layer_type = 'relu'):
    return ResNet18(num_classes=num_classes, norm_layer_type = norm_layer_type, conv_layer_type = conv_layer_type, linear_layer_type = linear_layer_type,
                    activation_layer_type = activation_layer_type)

# Simple model
def simpleCNN(num_classes = 10, norm_layer_type = 'bn' ,conv_layer_type = 'conv2d',linear_layer_type = 'linear', activation_layer_type = 'relu'):
    return CNN(num_classes=num_classes, norm_layer_type = norm_layer_type ,conv_layer_type = conv_layer_type,linear_layer_type = linear_layer_type,
               activation_layer_type = activation_layer_type)

def simpleCNN2(num_classes = 10, norm_layer_type = 'bn' ,conv_layer_type = 'conv2d',linear_layer_type = 'linear', activation_layer_type = 'relu'):
    return Net_circular_CNN(num_classes=num_classes, norm_layer_type = norm_layer_type ,conv_layer_type = conv_layer_type,linear_layer_type = linear_layer_type,
               activation_layer_type = activation_layer_type)
