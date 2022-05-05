"""
In this exercise you will port the edits that you did on the "exercise_2.py" script and train a copy of your previous
model on the FashionMNIST dataset, which is a MNIST-like dataset of 28*28 gray images of clothes, taken from Zalando
ads. This is to test the CNN on a slightly more challenging dataset. You should reuse the same training parameters, but
change the input normalization (MNIST and FashionMNIST have a differend std and mean).
Make a copy of the previous NN model and modify it in order to have a worst class accuracy of minimum 70%

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
- default model: size = 2.555MB, min class accuracy = 1.6, parameters = 669706, epochs = 5
- default optimized model: size = 2.555MB, min class accuracy = 68.40, parameters = 669706, epochs = 5
- optimized CNN model: size = 0.1233 MB, min class accuracy = 70.5, parameters = 32314, epochs = 5
The two default models, one trained without changing any parameter in this script and one trained by tuning the training
loop only (learning rate, data pre-processing) are provided in "saved_models", named exercise1_default.pth and
exercise1_default_optimized.pth respectively. The optimized CNN model is provided for reference and can be found in
"saved_models" as "exercise2_cnn.pth"
"""

import torch, torchvision, copy
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
from torch_neural_networks_library import mini_resnet
from pathlib import Path
from find_num_workers import find_num_workers
from torch.utils.tensorboard import SummaryWriter

Path("./runs/exercise_2bis").mkdir(parents=True, exist_ok=True)  # check if runs directory for tensorboard exist, if not create one

writer = SummaryWriter('runs/exercise_2bis')

'''
transform_train = transforms.Compose([transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])

training_data = datasets.FashionMNIST(root="data", train=True, download=False, transform=transform_train)
test_data = datasets.FashionMNIST(root="data", train=False, download=False, transform=transform_test)

batch = 16
best_workers = 6

train_dataloader = DataLoader(training_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=False)
test_dataloader = DataLoader(test_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=False)

mean = 0.
std = 0.
for images, _ in train_dataloader:
    batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(dim=2).sum(0)
    std += images.std(dim=2).sum(0)

for images, _ in test_dataloader:
    batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(dim=2).sum(0)
    std += images.std(dim=2).sum(0)

mean /= (len(train_dataloader.dataset) + len(test_dataloader.dataset))
std /= (len(train_dataloader.dataset) + len(test_dataloader.dataset))
print("Mean: ", mean, "Std: ", std)
'''

#transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2862], std=[0.3204]), transforms.RandomResizedCrop(size=(28,28), scale=(0.8, 1.0)), transforms.RandomHorizontalFlip()])
transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2862], std=[0.3204]), transforms.RandomHorizontalFlip()])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2862], std=[0.3204])])

training_data = datasets.FashionMNIST(root="data", train=True, download=False, transform=transform_train)
test_data = datasets.FashionMNIST(root="data", train=False, download=False, transform=transform_test)

#best_workers = find_num_workers(training_data=training_data, batch_size=batch_size)
best_workers = 6

'''
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
dataiter = iter(copy.deepcopy(test_dataloader))
images, labels = dataiter.next()
img_grid = torchvision.utils.make_grid(images)
writer.add_image(str(batch_size)+'_FashionMNIST_images', img_grid)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
'''
device = "cpu"

# TODO: write your own model in torch_neural_networks_library.py and call it here
model = mini_resnet()  # create model instance, initialize parameters, send to device

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
print(model)
'''
writer.add_graph(model, images)
model.to(device)
# Used to debugging summary(), delete if you want.
'''
params = sum([np.prod(p.size()) for p in model_parameters])
memory = params * 32 / 8 / 1024 / 1024
print("this model has ", params, " parameters")
print("total weight memory is %.4f MB" %(memory))

loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([2.,1.,2.2,1.1,2.,1.,5.,1.,1.,1.]))

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

        if batch % 100 == 0:
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

batch_size = [16, 32, 64, 128]
lr = 5e-3
epochs = 5
best_correct = 0
Path("./saved_models").mkdir(parents=True, exist_ok=True)
print("Use $ tensorboard --logdir=runs/exercise_2bis to access training statistics")
for batch in batch_size:
    model = mini_resnet()
    model.to(device)
    train_dataloader = DataLoader(training_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=False)
    test_dataloader = DataLoader(test_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=False)
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.00001, momentum=.8, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.6)
    print(f"using: batch={batch}, n_ep={epochs}, lr={lr}")
    for t in tqdm(range(epochs)):
        print(f"Epoch {t+1}\n-------------------------------")
        loss = train(train_dataloader, model, loss_fn, optimizer, t)
        current_correct = test(test_dataloader, model, loss_fn)
        scheduler.step()
        writer.add_scalar('test accuracy', current_correct, t)
        writer.flush()
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
            }, "./saved_models/exercise2bis.pth")
    #break

writer.close()
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

###load the best model..
opt_model = torch.load("./saved_models/exercise2bis.pth")
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

print("Worst class accuracy is %.4f for class %s" %(min_correct[1], min_correct[0]))

default_score = (2.555/memory) * 0.2 + (lowest_class_accuracy/1.6) * 0.3 + (669706.0/params) * 0.3 + (5/epochs) * 0.2
optimized_score = (2.555/memory) * 0.2 + (lowest_class_accuracy/68.40) * 0.3 + (669706.0/params) * 0.3 + (5/epochs) * 0.2
opt_CNN_score = (0.1233/memory) * 0.2 + (lowest_class_accuracy/70.5) * 0.3 + (32314.0/params) * 0.3 + (5/epochs) * 0.2


print("Score for this exercise against default model from exercise 1 = %.4f" %(default_score))
print("Score for this exercise against optimized training script from exercise 1 = %.4f" %(optimized_score))
print("Score for this exercise against optimized CNN and script from this exercise = %.4f" %(opt_CNN_score))

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

