"""
This is an introductory script with the basic functions and definitions required to implement the training loop of a
simple neural network for MNIST. For this script you do not have to change the code, just read and understand it.

The MNIST dataset consist of 60000 train images and 10000 test images of digits and was used to develop the first
convolutional neural network for handwritten digit recognition
http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

To execute this script you need:
- a conda installation and environment
- this script

You can use any IDE you want to edit, debug and run python scripts. In our team we use Pycharm Professional, which
license is free for university students. Go on this website https://www.jetbrains.com/pycharm/, make a student account
using your @studenti.polito.it mail. Download, install the IDE and activate the license.

Download Anaconda from this website, choose the version that suits your operating system and install it.
https://www.anaconda.com/products/individual

To install the environment type the following commands in a terminal, minus the $:
$ conda create -n cnn_exercises
$ conda activate cnn_exercises

if you don't have a GPU, you have to install the CPU version of Pytorch
$ conda install pytorch torchvision torchaudio cpuonly -c pytorch

if you have a NVIDIA GPU, please check you cuda driver version and install the appropriate version of pytorch.
$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

then, execute the following commands to install the remaining packages:
$ pip install brevitas
$ conda install tqdm
$ conda install -c conda-forge onnx
$ pip3 install onnxoptimizer
$ conda install -c conda-forge tensorflow

Tensorflow is needed to use tensorboard, which can not be used as a standalone package due to some bugs in the built-in
torch i/o functions.

Run this script once to download the MNIST dataset, it will be placed in the same directory of this
script, then make sure to change the option "download=True," to "download=False," on the two MNIST
instances, otherwise you might download and overwrite the entire dataset each time you run this script.
$ python exercise_0.py
"""

"""These are the import declarations, we will use these packages from exercise 0 to 3"""
import torch, torchvision, copy
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
from torch_neural_networks_library import default_model  # NN models will be defined in this python file
from pathlib import Path
from find_num_workers import find_num_workers

from torch.utils.tensorboard import SummaryWriter

Path("./runs/exercise_0").mkdir(parents=True, exist_ok=True)  # check if runs directory for tensorboard exist, if not create one

writer = SummaryWriter(log_dir='runs/exercise_0')

"""
A training script is essentially composed of 3 elements: datatest pipeline, training function and test function.
Dataset pipeline:  loads the data, apply pre-processing functions such as normalization, padding, shuffling, cropping
                   data augmentation, split and so on. The dataset pipeline is necessary to make all the input data 
                   coherent (for instance, same dimension) and feed it to the processing engines to avoid bottlenecks 
                   during the training process. 
Training function: for all the elements wrapped in a single batch, executes a forward pass, computes the error between 
                   the neural network predictions and the ground truth with a loss function, back propagates the error 
                   and evaluate the gradient, then updates the weight. It also returns the accuracy and loss.
Test function:     for all the elements wrapped in a single batch, execute a forward pass, counts or evaluates the 
                   correct predictions, returns the accuracy and loss.
"""

# TODO: Check transforms class for the methods documentation to understand the syntax and which pre-processing functions
#       are already implemented in pytorch. The same functions are also implemented in Tensorflow
transform = transforms.Compose([transforms.ToTensor()])

# TODO: set download to True if you are running this script for the first time, otherwise set it to False
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=False,   # set to false if the dataset has been downloaded already
    transform=transform,
)

# TODO: set download to True if you are running this script for the first time, otherwise set it to False
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=False,   # set to false if the dataset has been downloaded already
    transform=transform,
)


"""
The batch size sets how many inputs are processed at once during each forward and backward propagation pass. 
For each image, the framework needs to store additional data, thus allocate additional memory. High batch values improve
the training speed, but require more memory. Keep in mind that also model parameters during the training require memory
allocation, thus large neural networks require a lot of memory to be trained even with small batch sizes. 
"""
batch_size = 16

"""
Now it is possible to define the dataloader functions, which will wrap the previous dataset objects definitions, load 
the data from the memory to the system memory, then apply pre-processing on CPU, and finally move it to the device that
executes the training and test processes. Dataloaders, if not appropriately tuned, are often a bottleneck in the
training process. To do so, we run a benchmark to check. You can take note of this value and comment the following 
function for the next runs.
"""
#best_workers = find_num_workers(training_data=training_data, batch_size=batch_size)
best_workers = 6   # change this number with the one from the previous function and keep using that for future runs

#cuda_avail = torch.cuda.is_available()
cuda_avail = False

# TODO: Look into the DataLoader class and see which options can be selected to load the data.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=cuda_avail)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=cuda_avail)


for X, y in test_dataloader:  # Print the shape of the input and output data
    print("Shape of X [N, C, H, W]: ", X.shape, X.dtype)
    print("Shape of y: ", y.shape, y.dtype)
    break
print(test_data.classes)
"""
A popular way to check the training info and visualize the computational graph and other statistics is Tensorboard.
Here in this script there is already a writer function initialized, now we use it to visualize some images from the 
dataset.
"""

dataiter = iter(copy.deepcopy(test_dataloader))
images, labels = dataiter.next()
img_grid = torchvision.utils.make_grid(images)
writer.add_image(str(batch_size)+'_mnist_images', img_grid)

"""
Open a new terminal inside the folder
"""

#device = "cuda" if torch.cuda.is_available() else "cpu"  # Set cpu or gpu device for training.
device = "cpu"
print("Using {} device".format(device))

""" 
Notice how there is no explicit parameter initialization. How are initialized the learnable parameters? 
Why is it important to initialize parameters in a proper manner and not by just setting them to 1, 0 or random values?
Here is the list of methods for initialization https://pytorch.org/docs/stable/nn.init.html
"""
# TODO: move inside the model and then the layer instances, understand how layers are initialized
model = default_model()  # create model instance, initialize parameters, send to device
print(model)
writer.add_graph(model, images)
writer.flush()
"""It is necessary that both the model and the tensors involved in the inference/training are on the same device"""
model.to(device)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])   # count learnable parameters
memory = params * 32 / 8 / 1024 / 1024  # evaluate total parameter memory
print("this model has ", params, " parameters")
print("total weight memory is %.4f MB" %(memory))

"""
Loss function and optimizer definitions. Enter the CrossEntropyLoss and optim calls to see the methods already 
implemented in Pytorch. For these exercise we will use the Cross Entropy Loss, since we will be dealing with N-classes. 
You can find the loss function and optimizers description, mathematical formulas and documentation on the implementation 
at the following links.
Loss functions -> https://pytorch.org/docs/stable/nn.html#loss-functions
Optimizers -> https://pytorch.org/docs/stable/optim.html#algorithms
"""
# TODO: Read the L1loss, CrossEntropyLoss, SGD, RMSprop and Adam, since those are the ones you found in the CNN course.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5e-4)

def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward propagation
        pred = model(X)  # Compute prediction error
        loss = loss_fn(pred, y)  # Compute loss

        # Backpropagation
        optimizer.zero_grad()  # prepares the gradient by setting it to zero for all tensors
        loss.backward()  # computes the gradient of the graph, in this case the collection of layers
        optimizer.step()  # performs a parameter update using the method instantiated before, in this case the SGD

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            writer.add_scalar('training loss', loss / 1000, epoch * len(dataloader) + batch)
    return loss

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

""" The epochs value defines how many times the training loop is run over the entire dataset. The epoch size can be
tuned to select the minimum number that allows the model to converge to a stable train/test accuracy. Values too small 
or too big can lead to under/over fitting"""
epochs = 5
best_correct = 0
best_model = []

print("Use $ tensorboard --logdir=runs/exercise_0 to access training statistics")
Path("./saved_models").mkdir(parents=True, exist_ok=True)

for t in tqdm(range(epochs)):
    print(f"Epoch {t+1}\n-------------------------------")
    loss = train(train_dataloader, model, loss_fn, optimizer, t)
    current_correct = test(test_dataloader, model, loss_fn)
    writer.add_scalar('test accuracy', current_correct, t)
    torch.save({
        'epoch': t,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'test_acc': current_correct,
    }, "./saved_models/default_model.pth")
    print("Saved PyTorch Model State to model.pth")

writer.close()
"""
Class accuracy test loop, checks how accurate the neural network is for each class. The standard test loop only tests
for the overall accuracy, thus you do not really know how accurate is your model on each class. This might lead to a 
mis-interpretation of the real task accuracy, with some models that might have a really high accuracy on a subset 
of the dataset classes, while under-performing on others, which might result on a high overall test accuracy. 
This is important as it shows if the model has been trained enough or is enough complex for difficult cases. 
Unfortunately, we can not know hard classes before running these tests, nor we know if these classes are always the most 
difficult ones in different models. With this model and these default training parameters, you should get around 70% 
test accuracy, with two classes below 50% accuracy, one of which below 2-3% accuracy (almost one order of magnitude
below random guess, which is 10% for 10 classes). In the next exercise, you will improve the minimum class accuracy by
simply changing the dataset pipeline and training parameters.
"""
classes = test_data.classes

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for X, y in test_dataloader:
        images, labels = X.to(device), y.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

min_correct = [0,110]
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    if min_correct[1] >= int(accuracy):
        min_correct = [classname, accuracy]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))

lowest_class_accuracy = min_correct[1]

print("Worst class accuracy is %.4f for class %s" %(min_correct[1], min_correct[0]))

