import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import kornia
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding.

    Args:
        in_planes : Number of input channels.
        out_planes : Number of output channels (filters).
        stride: Stride of the convolution. Default is 1.
    
    Returns:
        nn.Conv2d: A 2D convolutional layer with 3x3 kernel size.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias = False)

def conv1x1(in_planes, out_planes, stride =1):
    """
    1x1 convolution.

    Args:
        in_planes: Number of input channels.
        out_planes: Number of output channels (filters).
        stride: Stride of the convolution. Default is 1.
    
    Returns:
        nn.Conv2d: A 2D convolutional layer with 1x1 kernel size.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride = stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1 # Expansion factor for output channels used in ResNet blocks
    num_layers = 2  # Number of layers in the block 

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Initialize a BasicBlock instance.
    
        Args:
            inplanes : Number of input channels.
            planes : Number of output channels for the convolutions.
            stride: Stride for the first convolution. Default is 1.
            downsample : Downsampling layer to match dimensions of the input and output for the residual connection.
        """
        super(BasicBlock, self).__init__()
        # only conv with possibly not 1 stride
        self.conv1 = conv3x3(inplanes, planes, stride)
        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(planes)
        # ReLU activation 
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # if stride is not 1 then self.downsample cannot be None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass for the BasicBlock.

        Args:
            x : Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the block.
        """
        identity = x
         # First convolution, batch normalization, and ReLU activation
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
         # Second convolution and batch normalization
        output = self.conv2(output)
        output = self.bn2(output)

        # Apply downsampling to the identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # the residual connection
        
        output += identity
        output = self.relu(output)

        return output

    def block_conv_info(self):
        """
        Retrieve information about the convolutional layers in the block.
        
        Returns:
            tuple: information about the convolutional layers
        """
        block_kernel_sizes = [3, 3]
        block_strides = [self.stride, 1]
        block_paddings = [1, 1]

        return block_kernel_sizes, block_strides, block_paddings

class ResNet_features(nn.Module):
    '''
    the convolutional layers of ResNet
    the average pooling and final fully convolutional layer is removed
    '''

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, filter=None, filter_layer=None):
        """
        Initialize ResNet_features.

        Args:
            block: The residual block type
            layers: Number of blocks in each layer
            num_classes: Number of classes for classification.
            zero_init_residual: Whether to zero-initialize residual batch norm weights.
            filter: Custom filter.
            filter_layer: Custom filter layer.
        """
        super(ResNet_features, self).__init__()
        self.inplanes = 64

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        # Max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Global average pooling
        self.global_pool = nn.AvgPool2d(kernel_size=7)


        # Define the sequential layers using residual blocks
        self.block = block
        self.layers = layers

        # comes from the first conv and the following max pool
        self.kernel_sizes = [7, 3]
        self.strides = [2, 2]
        self.paddings = [3, 1]

        # Layer 1
        self.layer1 = self._make_layer(block=block, planes=64, num_blocks=self.layers[0])
        self.dn_layer1 = None
        
        # Layer 2
        self.layer2 = self._make_layer(block=block, planes=128, num_blocks=self.layers[1], stride=2)
        self.dn_layer2 = None

        # Layer 3
        self.layer3 = self._make_layer(block=block, planes=256, num_blocks=self.layers[2], stride=2)
        self.dn_layer3 = None

        # Layer 4
        self.layer4 = self._make_layer(block=block, planes=512, num_blocks=self.layers[3], stride=2)
        self.dn_layer4 = None

         # Final fully connected layer
        self.fc = nn.Linear(512, 10)

        # initialize the weights
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        if zero_init_residual:
            for layer in self.modules():
                nn.init.constant_(layer.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        """
        Create a sequential layer consisting of multiple residual blocks.

        Args:
            block: The block type.
            planes : Number of output channels for the blocks.
            num_blocks: Number of blocks in the layer.
            stride: Stride for the first block. 

        Returns:
            nn.Sequential: A sequence of residual blocks.
        """
        downsample = None
        # Downsample for dimension matching when stride > 1 or channel mismatch
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
          # First block with optional downsampling
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        #  Track kernel, stride, and padding info
        for each_block in layers:
            block_kernel_sizes, block_strides, block_paddings = each_block.block_conv_info()
            self.kernel_sizes.extend(block_kernel_sizes)
            self.strides.extend(block_strides)
            self.paddings.extend(block_paddings)

        return nn.Sequential(*layers)

    def forward(self, inputs):
        """
        Standard forward pass through the network.
        """
        inputs = self.conv1(inputs)
        inputs = self.bn1(inputs)
        inputs = self.relu(inputs)
        inputs = self.maxpool(inputs)

        inputs = self.layer1(inputs)

        if self.dn_layer1 is not None:
            inputs = self.dn_layer1(inputs)
        inputs = self.layer2(inputs)
        if self.dn_layer2 is not None:
            inputs = self.dn_layer2(inputs)
        inputs = self.layer3(inputs)
        if self.dn_layer3 is not None:
            inputs = self.dn_layer3(inputs)
        inputs = self.layer4(inputs)
        if self.dn_layer4 is not None:
            inputs = self.dn_layer4(inputs)
        inputs = self.global_pool(inputs)
        # inputs = torch.flatten(inputs, 1) 
        inputs = inputs.reshape(-1, 512)

        output = self.fc(inputs)
        return output
    

def resnet_18(pretrained=False, **kwargs):
    """
    Constructs a ResNet-18 model.
    
    Args:
        pretrained: If True, returns a model pre-trained
    """
    model = ResNet_features(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model