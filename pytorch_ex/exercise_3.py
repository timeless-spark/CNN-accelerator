"""
In this exercise you will write a CNN model for CIFAR10.

CIFAR10 is a small dataset with 50000/10000 train/test images of 32*32*3 pixels with 10 classes.
In the next exercise, you will also experiment with CIFAR100.
Please check the following paper if you want to know more about the dataset
> https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

For this challenge you should look at different state-of-the-art convolutional neural networks, you can check the ones
included as reference in "layer_templates_pytorch.py"

You will notice that all the CNNs referenced use skip connections, which propagate the input activations without
performing any computation that are eventually added to the output of a sequence of convolutional layers.
The purpose of the skip connection is to solve the vanishing gradient effect, which prevents the training of very deep
models (the gradient is higher at the last layer and smaller at the first one, for deep model the gradient tends to zero
preventing any variable updates). You can read more on the problems of vanishing gradients and the purpose of the
residual block in the following paper, also referenced in "layer_templates_pytorch.py"
https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf

You should also consider using normalization layers, which normalize the input data during the training (a normalization
layer has learnable parameters) and also during the inference on new data.
Generally, normalization layers are necessary to build a high accuracy deep neural network, as they are necessary to
speed-up the training process by relaxing learning-rate and initialization constraints.
However, you should use the same initialization approach that you used in the previous exercises. If not, the default
pytorch initialization is already similar to the most commonly used approach (Glorot initialization).
More info on the use of batch normalization layers can be found here -> https://arxiv.org/pdf/1502.03167.pdf
Normalization layers are also used to prevent high magnitude scaling factors, which is important for quantized task
accuracy and adversarial attack / fault resilience.

Rules:
- You are free to do everything that you want
- The goal is still to write a model that has the best tradeoff between accuracy, model parameters and model size.
- For this task, you will compete against a small custom model trained on an ultrabook CPU, a deeper one trained on
  a RTX 3090, and a ResNet20 trained exactly as described in the paper "Deep Residual Learning for Image Recognition",
  namely exercise3_cpu.pth and exercise3_gpu.pth, and exercise3_gpu_resnet20.pth


- The score is evaluated as: (default model size/your model size) * A +
                             (your model min class accuracy/default model min accuracy +
                              your model test accuracy / default model test accuracy) * B +
                             (default model parameters/your model parameters) * C +
                             (default epochs/your epochs) * D

- The values of A, B and C are A = 0.2, B = 0.3, C = 0.3, D= 0.2
- cpu model: size = 0.3965MB, min class accuracy = 73.9, test accuracy = 86.9, parameters = 103946, epochs = 100
- gpu model: size = 1.1747MB, min class accuracy = 78.6, test accuracy = 89.8, parameters = 307946, epochs = 320
- ResNet20: size = 1.1205MB, min class accuracy = 77.6, test accuracy = 89.2, parameters = 293738, epochs = 320
The cpu model was trained on a 8-core Intel laptop for 100 epochs (around 3 hours), batch size 32
The gpu models were trained on a Nvidia RTX3090 for 320 epochs (around 50 minutes), batch size 256
"""

from telnetlib import NOP
import torch, torchvision, copy
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
from torch_neural_networks_library import isaResnet_14, isaResnet_26, isaResnet_50, isaResnet_98
from torch_neural_networks_library import ex3ResNet
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
Path("./runs/exercise_3").mkdir(parents=True, exist_ok=True)  # check if runs directory for tensorboard exist, if not create one

writer = SummaryWriter('runs/exercise_3')

PATH = "./saved_models/exercise3.pth"

load_pretrained = False  # For this exercise you might need to interrupt and resume long training sessions, use this
                         # flag to load a pre-trained model


transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), transforms.RandomResizedCrop(size=(32,32), scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(0.5)])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
# TODO: use two different transforms for train and test, both should apply input normalization, but only the training
#       transform should augment the data.
training_data, validation_data = random_split(datasets.CIFAR10(root="data", train=True, download=True, transform=transform_train), [45000, 5000])
test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)

best_workers = 2
batch_size = 32

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())

print("train dataset samples: ", len(train_dataloader.dataset))
print("validation dataset samples: ", len(validation_dataloader.dataset))
print("test dataset samples: ", len(test_dataloader.dataset))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
device = "cpu"

# TODO: write your own model in torch_neural_networks_library.py and call it here
model = ex3ResNet()  # create model instance, initialize parameters, send to device

model_parameters = filter(lambda p: p.requires_grad, model.parameters())

'''
writer.add_graph(model, images)
model.to(device)
# Used to debugging summary(), delete if you want.
'''
params = sum([np.prod(p.size()) for p in model_parameters])
memory = params * 32 / 8 / 1024 / 1024
print("this model has ", params, " parameters")
print("total weight memory is %.4f MB" %(memory))
'''
for name, par in model.named_parameters():
    print(name)
    print(par.size().numel())
'''
loss_fn = nn.CrossEntropyLoss()

start_epoch = 0

###unlock then...
'''
if load_pretrained:
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint['epoch'] + 1
    epochs += start_epoch
    last_loss = checkpoint['loss']
    last_test = checkpoint['test_acc']
    writer.add_scalar('training loss', last_loss / 1000, start_epoch * len(train_dataloader) )
    print("loading pre-trained model with %.4f train loss, %.4f test accuracy, trained for %d epochs" %(last_loss, last_test, start_epoch+1))
'''

def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            writer.add_scalar('training loss', loss / 100, epoch * len(dataloader) + batch)
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

###training parameters
batch_size = 256
lr = 8e-3
L2_lambda = 5e-6
wd = L2_lambda/lr
epochs = 60
opt_mil = [30,45,55]

best_correct = 0
best_model = []
Path("./saved_models").mkdir(parents=True, exist_ok=True)
print("Use $ tensorboard --logdir=runs to access training statistics")

model = ex3ResNet()
model.to(device)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())

### TRAINING THE ALL PARAMS..
params = model.parameters()
'''
### TRAINING ONLY THE BATCH NORM..
params = list()
for mod in model.modules():
    if isinstance(mod, nn.BatchNorm2d):
        params += list(mod.parameters())
'''
'''
### TRAINING BATCH NORM + FC LAYER..
params = list()
for mod in model.modules():
    if isinstance(mod, nn.BatchNorm2d):
        params += list(mod.parameters())
    if isinstance(mod, nn.Linear):
        params += list(mod.parameters())
'''
optimizer = torch.optim.SGD(params, weight_decay=wd, momentum=.8, lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt_mil, gamma=0.2, verbose=True)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, threshold=0.0001, threshold_mode='abs')

mil_index = 0
for t in tqdm(range(epochs)):
    print(f"Epoch {t+1}\n-------------------------------")
    loss = train(train_dataloader, model, loss_fn, optimizer, t)
    current_correct = test(validation_dataloader, model, loss_fn)
    scheduler.step()
    #scheduler.step(loss)
    writer.add_scalar('test accuracy', current_correct, t)
    if (t+1) == opt_mil[mil_index]:
        new_lr = scheduler.get_last_lr()
        lr = new_lr
        mil_index += 1
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'test_acc': current_correct,
            'device': device,
            'model': model,
            'train_parameters': {'batch': batch_size, 'actual_epoch': t, 'lr': lr, 'wd': wd, 'opt_step': opt_mil}
        }, f"./saved_models/exercise3_{mil_index}.pth")
        print("Saved PyTorch Model State")

writer.close()

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

resnet_score = (1.1205/memory) * 0.2 + (lowest_class_accuracy/77.6 + current_correct/89.2) * 0.3 + (293738.0/params) * 0.3 + (320/epochs) * 0.2
cpu_score = (0.3965/memory) * 0.2 + (lowest_class_accuracy/73.9 + current_correct/86.9) * 0.3 + (103946.0/params) * 0.3 + (100/epochs) * 0.2
gpu_score = (1.1747/memory) * 0.2 + (lowest_class_accuracy/78.6 + current_correct/89.8) * 0.3 + (307946.0/params) * 0.3 + (320/epochs) * 0.2


print("Score for this exercise against cpu model = %.4f" % cpu_score)
print("Score for this exercise against gpu model = %.4f" % gpu_score)
print("Score for this exercise against ResNet20 model = %.4f" % resnet_score)

"""
Hints:
1- use a fast optimizer, you can either develop one from scratch or use one of those available in the torch library
2- a small learning rate is too slow at the beginning of the training process, a big one will not grant convergence as 
   the training progress. DO NOT ADJUST THE LEARNING RATE FOR EVERY OPTIMIZER, read carefully the papers or google it
3- avoid using too many linear layers, they are over-parametrized for this task, try using other layers
4- if necessary, use large filters in the first layers only
5- use less channels in the first layers, more channels in the last ones
6- template for CONV layer is nn.Conv2d(in_channels=..., out_channels=..., kernel_size=(...), stride=..., padding=..., bias =...)
   you need to define these parameters for each Conv2d instance, do not use default values even if are the same as yours
7- pay attention to the dimensioning of input-output spatial dimensions, for a single dimension (or 2 dimension in case
   of square images) the formula is out = floor( (in - kernel + 2 * padding) / stride ) + 1
"""


