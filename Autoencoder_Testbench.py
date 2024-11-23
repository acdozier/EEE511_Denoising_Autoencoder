# To make this run, you basically just need to install numpy, matplotlib 
# and torch (and torchvision) with cuda for the specified versions

import torch # Torch 2.2.1+cu118
import numpy as np # numpy 1.26.4
import matplotlib.pyplot as plt # matplotlib 3.9.2
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(device)

###############################################################################
## Define model here
from Autoencoder import autoencoder
model = autoencoder().to(device)

import os
working_directory = os.path.dirname(os.path.abspath(__file__))

# Name your model you'd like to use here and the model you'd like to save to (should be .pth)
model_load = "Model_v1.pth"
model_save = model_load
model_directory = os.path.join(os.path.dirname(working_directory), "Denoising Autoencoder\Models")
model_path = os.path.join(model_directory, model_load)


# Open model here if it exists, if it doesn't one will be created at the end of the training session to model_save
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Model loaded successfully.")
else:
    print(f"Model: {model_load} does not exist. Continuing without loading model.")

###############################################################################
import torchvision # torchvision 0.20.1
import torchvision.transforms as transforms # torchvision 0.20.1
from torch.utils.data import DataLoader # Torch 2.2.1+cu118
from PIL import Image # Pillow 11.0.0
from IPython.display import display # ipython 8.27.0
# Also I'm using spyder 5.5.1 as my IDE

import CustomMNIST

# Batch size specifies how many images to process at a time, will depend on the GPU VRAM
BATCH_SIZE = 200

# Define helpful transforms
to_PIL = transforms.ToPILImage()
transform = transforms.ToTensor()

# Obtain the training and test dataset respetively
train_ds = CustomMNIST.CustomMNIST(root='./data', train=True, download=True, transform = transform)
test_ds = CustomMNIST.CustomMNIST(root='./data', train=False, download=True, transform = transform)

# Process the datasets in dataloaders
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl =  DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# batch0_data, batch0_target = next(iter(train_dl))

# display(to_PIL(batch0_data[0]))
# display(to_PIL(batch0_target[0]))

###############################################################################
## Define training loop parameters (learning rate, loss function, ect.)
learning_rate = 1e-4
tolerance = 1e-5
verbose = True
iterations = 50

optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate, 
                             betas=(0.9, 0.999), 
                             eps=1e-08, 
                             weight_decay=0, 
                             amsgrad=False)

# Use MSE Loss
loss_fn = torch.nn.MSELoss()

###############################################################################
## Training Loop

import time
optimizer.zero_grad()

error = tolerance + 1
i = 0

if verbose:
    print("Iteration |   Error   |  Accuracy  |  Time (s)")
    tbeg = time.time()

# For storing the training and testing accuracies and loss
training_accuracies = np.zeros(iterations)
training_loss = np.zeros(iterations)
testing_accuracies = np.zeros(iterations)
testing_loss = np.zeros(iterations)

while error > tolerance and i < iterations:
    
    for batch, data in enumerate(train_dl):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    
    error = loss.item()
    
    
    if verbose and i%max(int(iterations * 0.01),1) == 0:
        tend = time.time()
        
        # Store accuracy and loss
        training_accuracies[i] = torch.sum(torch.abs(outputs - targets) < 1/32)/torch.numel(outputs)
        training_loss[i] = loss
        
        print(f"Train{i:<4}   {error:<6.3e}     {training_accuracies[i]:<6.4f}       {(tend - tbeg):.3f}")
        # Display some example training results
        display(to_PIL(inputs[0]))
        display(to_PIL(targets[0]))
        display(to_PIL(outputs[0]))
    
    for batch, data in enumerate(test_dl):
        test_inputs, test_targets = data
        test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
        
        test_outputs = model(test_inputs)
        test_loss = loss_fn(test_outputs, test_targets)
    
    test_error = test_loss.item()
    
    
    if verbose and i%max(int(iterations * 0.01),1) == 0:
        tend = time.time()
        
        # Store accuracy and loss
        testing_accuracies[i] = torch.sum(torch.abs(test_outputs - test_loss) < 1/32)/torch.numel(test_outputs)
        testing_loss[i] = test_loss
        
        print(f"Test{i:<4}    {test_error:<6.3e}     {testing_accuracies[i]:<6.4f}       {(tend - tbeg):.3f}")
        # Display some example training results
        display(to_PIL(test_inputs[0]))
        display(to_PIL(test_targets[0]))
        display(to_PIL(test_outputs[0]))
        
    i = i + 1

# ###############################################################################
# ## Save the model and display a small sample size

torch.save(model.state_dict(), os.path.join(model_directory, model_save))

# Send the model back to the cpu
model = model.cpu()

# ###############################################################################
# ## Plot the Training and Test Loss and Accuracies
plt.plot(np.arange(0,iterations), training_loss, label='Training Loss')
plt.plot(np.arange(0,iterations), testing_loss, label='Test Loss')
plt.legend()
plt.show()
plt.figure()
plt.plot(np.arange(0,iterations), training_accuracies, label="Training Accuracy")
plt.plot(np.arange(0,iterations), testing_accuracies, label="Test Accuracy")
plt.legend()
plt.show()
