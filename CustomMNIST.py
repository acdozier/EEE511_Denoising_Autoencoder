from torch.utils.data import Dataset # Torch 2.2.1+cu118
from torchvision import datasets, transforms # torchvision 0.20.1
import torch # Torch 2.2.1+cu118
import numpy as np # numpy 1.26.4

# Custom MNIST class to modify the MNIST data to have noisy images as inputs
# and original images as targets
class CustomMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        self.mnist = datasets.MNIST(root, train=train, transform=transform, download=download)
        self.target_transform = target_transform
        
        # Variance of the Noise
        self.variance = 0.4
        #self.variance = 0.8
    
    # Function uised for calling a member of the class
    def __getitem__(self, index):
        image, label = self.mnist[index]

        # The new target is the original image, while the new image is the
        # image with guassian/normal noise added to it
        new_target = image
        new_image = torch.clamp(image + torch.randn(image.shape) * np.sqrt(self.variance), min=0, max=1)

        return new_image, new_target

    def __len__(self):
        return len(self.mnist)
