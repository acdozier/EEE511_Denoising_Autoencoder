# Torch 2.2.1+cu118
import torch
import torch.nn as nn

#Convolution block with 3x3 kernel, same padding, relu activation, and batch normalization
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, batch=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
        if batch == True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity(out_channels)
    def forward(self, inputs):
        
        # Convolution, then relu, then batch normalization
        outputs = self.conv(inputs)
        outputs = self.relu(outputs)
        outputs = self.bn(outputs)
        return outputs

# Convollution block with 3x3 kernel, same padding, and sigmoid activation
class last_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.sig = nn.Sigmoid()

    def forward(self, inputs):
        
        # Convolution, then sigmoid
        outputs = self.conv(inputs)
        outputs = self.sig(outputs)
        return outputs

# Downscale block using larger stride convolution, 2x2 kernels, and no padding (Halves the size)
class downscale_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        
        # Downscaling convolution, then relu
        outputs = self.down(inputs)
        outputs = self.relu(outputs)
        return outputs

# Upscale block using transposed convolutions to increase size (2 times the size), 2x2 kernels and stride=2
class upscale_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
    
        # Upscaling convolution, then relu
        outputs = self.up(inputs)
        outputs = self.relu(outputs)
        return outputs

# Encoder block which uses 2 convolution blocks followed by a downscaling block
class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = conv_block(in_channels, in_channels)
        self.conv2 = conv_block(in_channels, in_channels)
        self.down = downscale_block(in_channels, out_channels)
    
    def forward(self, inputs):
        intermed = self.conv1(inputs)
        intermed = self.conv2(intermed)
        outputs = self.down(intermed)
        return outputs


# Decoder block which uses 2 convolution blocks followed by an upscaling block
class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = conv_block(in_channels, in_channels)
        self.conv2 = conv_block(in_channels, in_channels)
        self.up = upscale_block(in_channels, out_channels)

    def forward(self, inputs):
        intermed = self.conv1(inputs)
        intermed = self.conv2(intermed)
        outputs = self.up(intermed)
        return outputs
        
# Main class for the autoencoder
class autoencoder(nn.Module):
    #Class constructor
    def __init__(self):
        super().__init__()
        
        BASE_SIZE = 64
        
        #Define the encoder blocks
        self.start = conv_block(1, BASE_SIZE)
        self.e1 = encoder_block(BASE_SIZE, 2*BASE_SIZE)
        self.e2 = encoder_block(2*BASE_SIZE,4*BASE_SIZE)
        
        #Define the bottleneck block
        self.b = conv_block(4*BASE_SIZE, 4*BASE_SIZE)
        
        #Define the decoder blocks
        self.d1 = decoder_block(4*BASE_SIZE, 2*BASE_SIZE)
        self.d2 = decoder_block(2*BASE_SIZE, BASE_SIZE)

        # Define the last convolution block
        self.last = last_conv_block(BASE_SIZE, 1)
    
    def forward(self, inputs):
        
        # Encoding Stage
        en1 = self.start(inputs)
        en2 = self.e1(en1)
        en3 = self.e2(en2)
        
        # Bottleneck at the bottom
        bottleneck = self.b(en3)
        
        # Decoding Stage
        dec1 = self.d1(bottleneck)
        dec2 = self.d2(dec1)
        
        # Final Step
        outputs = self.last(dec2)
        return outputs
        
