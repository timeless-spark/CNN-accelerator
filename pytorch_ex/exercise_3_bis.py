'''
    Training only Batch Normalization layers version..
    
'''

import torch, torchvision, copy
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
from torch_neural_networks_library import isaResNet_14, isaResNet_56, isaResNet_110, isaResNet_110_normal, isaResNet_110_sparse_50, isaResNet_110_sparse_80, isaResNet_110_dropout, isaResNet_290
from pathlib import Path

#base_path = "../../drive/MyDrive/"
base_path = "./"

Path(base_path + "saved_models").mkdir(parents=True, exist_ok=True)

initialize_dict = False

transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), transforms.RandomResizedCrop(size=(32,32), scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(0.5)])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
training_data, validation_data = random_split(datasets.CIFAR10(root="data", train=True, download=True, transform=transform_train), [45000, 5000])
test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)

best_workers = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
#device = "cpu"

model_list = [isaResNet_110, isaResNet_110_normal, isaResNet_110_sparse_50, isaResNet_110_sparse_80, isaResNet_110_dropout]

def return_model_params(model):
    '''
    ### TRAINING THE ALL PARAMS..
    params = model.parameters()
    '''
    
    ### TRAINING ONLY THE BATCH NORM..
    params = list()
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            params += list(mod.parameters())
    
    '''
    ### TRANING BATCH NORM + DOWNSAMPLE CONV..
    params = list()
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            params += list(mod.parameters())
        if isinstance(mod, nn.Conv2d) and mod.kernel_size == (1, 1) and mod.stride == (2, 2):
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
    return params

def train(dataloader, model, loss_fn, optimizer, loss_list):
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

        if batch % 60 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            loss_list.append(loss / 100)
    return loss

def test(dataloader, model, loss_fn, loss_list=None, verbose=True):
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
    if verbose:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return correct

###training parameters (according to the paper..)
batch_size = 128
lr = 1e-1
L2_lambda = 1e-5
wd = L2_lambda/lr
epochs = 160
loss_fn = nn.CrossEntropyLoss()
#loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.,0.8,1.25,1.25,1.25,1.2,1.2,1.,1.,1.]).to(device))

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())

print("train dataset samples: ", len(train_dataloader.dataset))
print("validation dataset samples: ", len(validation_dataloader.dataset))
print("test dataset samples: ", len(test_dataloader.dataset))

if initialize_dict:
    tr_dict = {}
    for func in model_list:
        model, name = func()
        params = return_model_params(model)
        optimizer = torch.optim.SGD(params, weight_decay=wd, momentum=0.9, lr=lr)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, threshold=1e-4, threshold_mode='abs', verbose=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120], gamma=0.1)
        tr_dict[name] = {
            "model_state_dict": [model.state_dict(), 0],
            "learnable_params": sum(torch.numel(p) for p in return_model_params(model)),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch_done": 0,
            "training_loss": [],
            "validation_loss": [],
            "validation_acc": []
        }
    torch.save(tr_dict, base_path + "saved_models/exercise3_bis.pth")

tr_dict = torch.load(base_path + "saved_models/exercise3_bis.pth")

for func in model_list:
    model, name = func()
    if tr_dict[name]["epoch_done"] < epochs:
        print("training model: ", name)
        model.load_state_dict(tr_dict[name]["model_state_dict"][0])
        model.to(device)
        params = return_model_params(model)
        optimizer = torch.optim.SGD(params, weight_decay=wd, momentum=0.9, lr=lr)
        optimizer.load_state_dict(tr_dict[name]["optimizer_state_dict"])
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, threshold=1e-4, threshold_mode='abs', verbose=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120], gamma=0.1, verbose=True)
        scheduler.load_state_dict(tr_dict[name]["scheduler_state_dict"])
        if tr_dict[name]["epoch_done"] == 0:
          best_acc = 0
        else:
          best_acc = tr_dict[name]["model_state_dict"][1]

        for t in tqdm(range(epochs - tr_dict[name]["epoch_done"])):
            print(f"Epoch {t+1}\n-------------------------------")
            loss = train(train_dataloader, model, loss_fn, optimizer, tr_dict[name]["training_loss"])
            current_acc = test(validation_dataloader, model, loss_fn, tr_dict[name]["validation_loss"])
            #scheduler.step(tr_dict[name]["validation_loss"][-1])
            scheduler.step()
            tr_dict[name]["validation_acc"].append(current_acc * 100)
            tr_dict[name]["epoch_done"] += 1
            tr_dict[name]["optimizer_state_dict"] = optimizer.state_dict()
            tr_dict[name]["scheduler_state_dict"] = scheduler.state_dict()
            if current_acc > best_acc:
                best_acc = current_acc
                tr_dict[name]["model_state_dict"] = [model.state_dict(), best_acc]
            torch.save(tr_dict, base_path + "saved_models/exercise3_bis.pth")
            print(f"lr: {optimizer.param_groups[0]['lr']:0.2e}\n")

###-----

import numpy as np
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

tr_dict = torch.load(base_path + "saved_models/exercise3_bis.pth", map_location=torch.device('cpu'))

model_list = [(isaResNet_110, "isaResNet_110_downsample_loss"), (isaResNet_56, "isaResNet_56_downsample_loss"), (isaResNet_290, "isaResNet_290_downsample_loss")]

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']

Path(base_path + "ex3bis_figures").mkdir(parents=True, exist_ok=True)

for func in model_list:
    (model, _), name = func[0](), func[1]
    print(name, ":")
    print("\tLearnable parameters: {:d}".format(tr_dict[name]["learnable_params"]))
    print("\tValidation Accuracy: {:.1f} %".format(tr_dict[name]["model_state_dict"][1] * 100))
    model.load_state_dict(tr_dict[name]["model_state_dict"][0])
    model.to(device)

    #training loss
    step = len(tr_dict[name]["training_loss"]) / epochs
    n_ele = len(tr_dict[name]["training_loss"])
    n_epoch = tr_dict[name]["epoch_done"]
    start = n_epoch / n_ele
    tr_x_axis = np.linspace(start, n_epoch, n_ele)
    tr_y_axix = np.empty_like(tr_x_axis)
    i = 0
    for val in tr_dict[name]["training_loss"]:
        tr_y_axix[i] = val * 100
        i += 1

    #validation loss during training
    val_x_axis = np.linspace(1, tr_dict[name]["epoch_done"], tr_dict[name]["epoch_done"])
    val_y_axix = np.empty_like(val_x_axis)
    i = 0
    for val in tr_dict[name]["validation_loss"]:
        val_y_axix[i] = val
        i += 1

    plt.figure(figsize=[6.4, 4.8])
    plt.plot(tr_x_axis, tr_y_axix, linewidth=2.0, label="Training loss")
    plt.plot(val_x_axis, val_y_axix, linewidth=3.0, label="Validation loss")
    plt.ylim(top=3)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training loss", "Validation loss"])
    plt.savefig(base_path + f"ex3bis_figures/{name}_training_loss.png")
    plt.clf()

    #validation accuracy during training
    x_axis = np.linspace(1, tr_dict[name]["epoch_done"], tr_dict[name]["epoch_done"])
    y_axix = np.empty_like(x_axis)
    i = 0
    for val in tr_dict[name]["validation_acc"]:
        y_axix[i] = val
        i += 1
    
    plt.plot(x_axis, y_axix, linewidth=3.0)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Validation accuracy"])
    plt.savefig(base_path + f"ex3bis_figures/{name}_validation_accuracy.png")
    plt.clf()

    #test accuracy
    accuracy = test(test_dataloader, model, loss_fn, verbose=False)
    print("\tTest Accuracy: {:.1f} %".format(accuracy * 100))

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

    x_axis = x = 0.5 + np.arange(10)
    y_axix = np.empty(len(classes))
    i = 0
    min_correct = [0,110]
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        if min_correct[1] >= int(accuracy):
            min_correct = [classname, accuracy]
        #print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
        y_axix[i] = accuracy
        i += 1
    
    plt.figure(figsize=[12.8, 4.8])
    plt.bar(x_axis, y_axix, color=colors, width=0.9, tick_label=classes)
    plt.ylabel("Accuracy")
    plt.savefig(base_path + f"ex3bis_figures/{name}_single_class_accuracy.png")
    plt.clf()

    print("\tMin per-class accuracy is %.4f for class %s\n" %(min_correct[1], min_correct[0]))