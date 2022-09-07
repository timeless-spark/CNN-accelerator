"""
The goal for this exercise is to write a neural network for MNIST that uses a maximum of 6 layers with learnable
parameters. You must write your own model inside "torch_neural_networks_library.py", selecting the layer type and the
dimensions of kernel, input and output volume.

The model we used up to now is rather simple, yet over-parametrized and slow to train and execute for a simple task like
digit recognition. This is because the neural network uses fully connected layers only. As you should know by now,
convolutional layers are particularly suited for image processing tasks, such as handwritten digits recognition.
Try to use at least two convolutional layers (conv2d) inside your model and apply all the optimizations that you have
learnt in the previous exercise.

Rules:
- You can adjust the batch size according to the memory capacity of your processing unit
- You can NOT change the optimizer, but you can change its parameters
- You can change the epoch size
- You can change the pre-processing functions
- You can fully customize the class NeuralNetwork, thus your CNN model
- You must use at most 6 layers with learnable parameters (do not use norm layers, they are not necessary and count as
  layers with learnable parameters, you will use them in the next exercises)
- The goal is to write a model that has the best tradeoff between accuracy, model parameters and model size.

- The score is evaluated as: (default model size/your model size) * A +
                             (your model min class accuracy/default model min accuracy) * B +
                             (default model parameters/your model parameters) * C +
                             (default epochs/your epochs) * D

- The coefficients are: A = 0.2, B = 0.3, C = 0.3, D= 0.2
- default model: size = 2.555MB, min class accuracy = 5.9417, parameters = 669706, epochs = 5
- default optimized model: size = 2.555MB, min class accuracy = 96.0357, parameters = 669706, epochs = 3
- optimized CNN model: size = 0.1229 MB, min class accuracy = 98.2161, parameters = 32218, epochs = 5
The two default models, one trained without changing any parameter in this script and one trained by tuning the training
loop only (learning rate, data pre-processing) are provided in "saved_models", named exercise1_default.pth and
exercise1_default_optimized.pth respectively. The optimized CNN model is provided for reference and can be found in
"saved_models" as "exercise2_cnn.pth"
"""

from asyncio import base_tasks
import torch, torchvision, copy
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
from torch_neural_networks_library import micro_resnet, nano_resnet, custom_2
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import time

Path("./runs/exercise_2").mkdir(parents=True, exist_ok=True)  # check if runs directory for tensorboard exist, if not create one

#transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1309], std=[0.3018]), transforms.RandomResizedCrop(size=(28,28), scale=(0.8, 1.0)), transforms.RandomPerspective(distortion_scale=0.5)])
transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1309], std=[0.3018])])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1309], std=[0.3018])])

training_data = datasets.MNIST(root="data", train=True, download=True, transform=transform_train)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform_test)

#best_workers = find_num_workers(training_data=training_data, batch_size=batch_size)
best_workers = 6

'''
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
print(test_data.classes)
dataiter = iter(copy.deepcopy(test_dataloader))
images, labels = dataiter.next()
img_grid = torchvision.utils.make_grid(images)
writer.add_image(str(batch_size)+'_mnist_images', img_grid)
'''
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

loss_fn = nn.CrossEntropyLoss()

def train(dataloader, model, loss_fn, optimizer, loss_list, batch_size):
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

        if batch % int(11600 / batch_size) == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            loss_list.append(loss / 100)
    return loss

def test(dataloader, model, loss_fn, loss_list=None):
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
    if loss_list is not None:
        loss_list.append(test_loss)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

batch_size = [12, 16, 32]
epochs = 5
lr = 1e-2

res_dict = {
    "nano_resnet": [],
    "micro_resnet": [],
    "custom_2": [],
}
name_list = ["nano_resnet", "micro_resnet", "custom_2"]
model_list = [nano_resnet, micro_resnet, custom_2]

index = 0
for model_type in model_list:
    model_parameters = filter(lambda p: p.requires_grad, model_type().parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    memory = params * 32 / 8 / 1024 / 1024
    for batch in batch_size:
        model = model_type()
        model.to(device)
        train_dataloader = DataLoader(training_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=False)
        test_dataloader = DataLoader(test_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=False)
        optimizer = torch.optim.SGD(model.parameters(), weight_decay=lr/128, momentum=.8, lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,4], gamma=0.2, verbose=True)
        best_correct = 0
        training_loss = []
        avg_loss = []
        start = time.time()
        for t in tqdm(range(epochs)):
            print(f"Epoch {t+1}\n-------------------------------")
            loss = train(train_dataloader, model, loss_fn, optimizer, training_loss, batch)
            current_correct = test(test_dataloader, model, loss_fn, avg_loss)
            scheduler.step()
            #writer.add_scalar('test accuracy', current_correct, t)
            #writer.flush()
            if current_correct > best_correct:
                best_correct = current_correct
                torch.save({
                    "batch_size": batch,
                    "lr": lr,
                    'epoch': t,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'test_acc': current_correct,
                }, "./saved_models/exercise2.pth")
        total_time = time.time() - start

        classes = test_data.classes

        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        ###load the best model..
        opt_model = torch.load("./saved_models/exercise2.pth")
        print(opt_model["test_acc"])
        model.load_state_dict(opt_model["model_state_dict"])

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

        #print("Worst class accuracy is %.4f for class %s" %(min_correct[1], min_correct[0]))

        default_score = (2.555/memory) * 0.2 + (lowest_class_accuracy/5.9417) * 0.3 + (669706.0/params) * 0.3 + (5/epochs) * 0.2
        optimized_score = (2.555/memory) * 0.2 + (lowest_class_accuracy/96.0357) * 0.3 + (669706.0/params) * 0.3 + (3/epochs) * 0.2
        opt_CNN_score = (0.1229/memory) * 0.2 + (lowest_class_accuracy/98.2161) * 0.3 + (32218.0/params) * 0.3 + (5/epochs) * 0.2

        res_dict[name_list[index]].append([batch, lr, opt_model["test_acc"], opt_model["loss"], lowest_class_accuracy, training_loss, avg_loss, total_time, default_score, optimized_score, opt_CNN_score])

    index += 1

Path("./saved_models").mkdir(parents=True, exist_ok=True)

torch.save(res_dict, "./saved_models/ex2_res.pth")

### AGIUNGERE PARTE NICOLA..

"""
Hints:
1- a small learning rate is too slow at the beginning of the training process, a big one will not grant convergence as 
   the training progress
2- avoid using too many linear layers, they are over-parametrized for this task, try using other layers
3- if necessary, use large filters in the first layers only
4- use less channels in the first layers, more channels in the last ones
5- template for CONV layer is nn.Conv2d(in_channels=..., out_channels=..., kernel_size=(...), stride=..., padding=..., bias =...)
   you need to define these parameters for each Conv2d instance, do not use default values even if are the same as yours
6- pay attention to the dimensioning of input-output spatial dimensions, for a single dimension (or 2 dimension in case
   of square images) the formula is out = floor( (in - kernel + 2 * padding) / stride ) + 1
"""

