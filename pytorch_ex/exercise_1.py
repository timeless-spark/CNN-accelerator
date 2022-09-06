"""
The goal of this exercise is to maximise the accuracy of a given neural network model optimizing the training setup.

Please check the TODO lines (in yellow if you are using PyCharm) to find the parts of code that you need to edit

Rules:
- You can NOT change the neural network model, but you can change the learnable parameters initialization
- You can adjust the batch size according to the memory capacity of your processing unit
- You can NOT change the optimizer, but you can change its parameters
- You can change the epoch size
- You can change the pre-processing functions
- You can not change the neural network model

- The goal is to write a model that has the best tradeoff between accuracy, model parameters and model size. You will
  compare the model performance against one trained with the default script and one trained with an optimized script

- The score is evaluated as: (your model min class accuracy/default model min accuracy) * A +
                             (default epochs/your epochs) * B
- The coefficients are: A = 0.6, B = 0.4
- default min class accuracy = 5.9417, default epochs = 5
- default optimized min class accuracy = 96.0357, default optimized epochs = 3

The two default models, one trained without changing any parameter in this script and one trained by tuning the training
loop only (learning rate, data pre-processing) are provided in "saved_models", named exercise1_default.pth and
exercise1_default_optimized.pth respectively.
"""

from pickle import FALSE
import torch, torchvision, copy
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
from torch_neural_networks_library import default_model
#from find_num_workers import find_num_workers
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import time

Path("./runs/exercise_1").mkdir(parents=True, exist_ok=True)  # check if runs directory for tensorboard exist, if not create one
writer = SummaryWriter('runs/exercise_1')

# TODO: which input pre-processing can you do to improve the accuracy? Check the transforms class to see the supported
#       input transformations, choose the most meaningful one for you and add it. Check transforms class for
#       documentation to understand the syntax.

'''
transform_train = transforms.Compose([transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])

training_data = datasets.MNIST(root="data", train=True, download=False, transform=transform_train)
test_data = datasets.MNIST(root="data", train=False, download=False, transform=transform_test)

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

#transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1309], std=[0.3018]), transforms.RandomResizedCrop(size=(28,28), scale=(0.8, 1.0)), transforms.RandomPerspective(distortion_scale=0.5)])
transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1309], std=[0.3018])])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1309], std=[0.3018])])

training_data = datasets.MNIST(root="data", train=True, download=True, transform=transform_train)
test_data = datasets.MNIST(root="data", train=False, download=False, transform=transform_test)

#best_workers = find_num_workers(training_data=training_data, batch_size=batch_size)
best_workers = 6   # change this number with the one from the previous function and keep using that for future runs

'''
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
print(test_data.classes)

dataiter = iter(copy.deepcopy(test_dataloader))
images, labels = dataiter.next()
img_grid = torchvision.utils.make_grid(images)
writer.add_image(str(batch_size[2])+'_mnist_images', img_grid)
'''

# Get cpu or gpu device for training.
#device = "cuda" if torch.cuda.is_available() else "cpu"
#print("Using {} device".format(device))

'''
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
print(model)
writer.add_graph(model, images)
model.to(device)
# Used to debugging summary(), delete if you want.
params = sum([np.prod(p.size()) for p in model_parameters])
memory = params * 32 / 8 / 1024 / 1024
print("this model has ", params, " parameters")
print("total weight memory is %.4f MB" %(memory))
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

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
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

loss_fn = nn.CrossEntropyLoss()

# TODO: check how weights are initialized and try to use different methods, to do this you need to add some lines of
#       code in the model definition. What are the best methods and why?
    #model defined in the loop......
device = "cpu"
print("Using {} device".format(device))

# TODO: Which parameters can you change in the Stochastic Gradient Descent optimizer? Is the default learning rate
#       appropriate for a fast convergence?
    #optimizer and scheduler in the loop......
lr_list = [5e-2, 5e-3]
lr_decay = [0.1, 0.5]
epoch_list = [[3,1], [5,2]]

# TODO: find the optimal batch size for your training setup. The batch size influences how much GPU or system memory is
#       required, but also influences how fast the optimizer can converge to the optima. Values too big or too
#       small will slow down your training or even cause a crash sometimes, try to find a good compromise. Use the
#       average loss and iteration time displayed in the console during the training to tune the batch size.
batch_list = [12, 16, 32]

# TODO: change the epochs parameter to change how many times the model is trained over the entire dataset. How many
#       epochs does your model require to reach the optima or oscillate around it? How many epochs does your model
#       require to get past 80% accuracy? How many for 90%? How can you speed-up the training without increasing the
#       epochs from the default value of 5?

results = []
index = 0

best_correct = 0
Path("./saved_models").mkdir(parents=True, exist_ok=True)
print("Use $ tensorboard --logdir=runs/exercise_1 to access training statistics")
for epoch in epoch_list:
    for batch in batch_list:
        for lr in lr_list:
            for gamma in lr_decay:
                model = default_model()
                model.to(device)
                train_dataloader = DataLoader(training_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=False)
                test_dataloader = DataLoader(test_data, batch_size=batch, num_workers=best_workers, pin_memory=False)
                optimizer = torch.optim.SGD(model.parameters(), weight_decay=lr/128, momentum=.8, lr=lr)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch[1], gamma=gamma)
                print(f"using: batch={batch}, n_ep={epoch[0]}, lr={lr}")
                start = time.time()
                for t in tqdm(range(epoch[0])):
                    print(f"Epoch {t+1}\n-------------------------------")
                    loss = train(train_dataloader, model, loss_fn, optimizer, t)
                    current_correct = test(test_dataloader, model, loss_fn)
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
                        }, "./saved_models/exercise1.pth")
                total_time = time.time() - start

                classes = test_data.classes

                correct_pred = {classname: 0 for classname in classes}
                total_pred = {classname: 0 for classname in classes}

                ###load the best model..
                opt_model = torch.load("./saved_models/exercise1.pth")
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

                default_score = (lowest_class_accuracy/5.9417) * 0.6 + (5/epoch[0]) * 0.4
                score = (lowest_class_accuracy/96.0357) * 0.6 + (3/epoch[0]) * 0.4
                results.append([batch, epoch[0], lr, gamma, opt_model["test_acc"], opt_model["loss"], lowest_class_accuracy, total_time, default_score, score])

                Path("./saved_models").mkdir(parents=True, exist_ok=True)

                res_dict = {"results": results}
                torch.save(res_dict, "./saved_models/ex1_res.pth")

                index += 1

torch.load(res_dict, "./saved_models/ex1_res.pth")

index = 0
for res in results:
    print("Case %d" % index)
    print("Batch size: %d" % res[0])
    print("Epoch: %d" % res[1])
    print("LR: %f" % res[2])
    print("Gamma: %.2f" % res[3])
    print("test_acc: %.4f" % res[4])
    print("avg_loss: %.4f" % res[5])
    print("lowest class: %.4f" % res[6])
    print("time: %d" % res[7])
    print("def score: %.4f" % res[8])
    print("opt score: %.4f\n\n" % res[9])
    index += 1

print("Worst class accuracy is %.4f for class %s" %(min_correct[1], min_correct[0]))
default_score = (lowest_class_accuracy/5.9417) * 0.6 + (5/epoch) * 0.4
score = (lowest_class_accuracy/96.0357) * 0.6 + (3/epoch) * 0.4

print("Score for this exercise against default training script = %.4f" %(default_score))
print("Score for this exercise against optimized training script = %.4f" %(score))

