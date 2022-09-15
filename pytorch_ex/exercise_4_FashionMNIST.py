from math import remainder
from pickletools import optimize
import torch, torchvision, copy
from torch import nn, true_divide
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
from exercise_4 import quant_custom_mini_resnet, quant_custom_mini_resnet_folded, quant_brevitas_mini_resnet, PACT_QuantReLU
import quantization as cs
from quantization import myDoReFaWeightQuantization
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import brevitas.config
from brevitas.export.onnx.finn.manager import FINNManager
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
import brevitas.nn as qnn
from brevitas.quant_tensor import QuantTensor 
import re

brevitas.config.IGNORE_MISSING_KEYS=True

base_path = "./"

Path(base_path + "saved_models").mkdir(parents=True, exist_ok=True)

transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2862], std=[0.3204]), transforms.RandomHorizontalFlip()])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2862], std=[0.3204])])

training_data, validation_data = random_split(datasets.FashionMNIST(root="data", train=True, download=True, transform=transform_train), [50000, 10000])
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform_test)

initialize_dict = True
best_workers = 2

device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
#device = "cpu"

#model_list = [(quant_custom_mini_resnet_folded, "quant_custom_mini_resnet_folded_8", 5e-3, 8, 8, 16, "scale"), (quant_custom_mini_resnet_folded, "quant_custom_mini_resnet_folded_16", 5e-3, 16, 16, 32, "scale")]
model_list = [(quant_custom_mini_resnet_folded, "quant_custom_mini_resnet_folded_4", 8e-4, 4, 4, 8, "scale")]
#model_list = [(quant_brevitas_mini_resnet, "quant_brevitas_mini_resnet", 5e-3)]

for p in model_list:
    print(p[1])

brevitas_model = False

def train(dataloader, model, loss_fn, optimizer, loss_list=None):
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

        if batch % 120 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            if loss_list is not None:
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
            if isinstance(pred, torch.Tensor):
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            else:
                correct += (pred.value.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    if loss_list is not None:
        loss_list.append(test_loss)
    if verbose:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return correct

loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.,0.5,1.1,0.55,1.,0.5,2.5,0.5,0.5,0.5]))

batch_size = 64
L2_lambda = 1e-9
epochs = 60

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())

print("train dataset samples: ", len(train_dataloader.dataset))
print("validation dataset samples: ", len(validation_dataloader.dataset))
print("test dataset samples: ", len(test_dataloader.dataset))

if initialize_dict:
    tr_dict = {}

    for model_type in model_list:
        if brevitas_model:
            model = model_type[0]()
        else:
            model = model_type[0](model_type[3], model_type[4], model_type[5], model_type[6])
        name = model_type[1]
        lr = model_type[2]
        wd = L2_lambda/lr
        params = model.parameters()
        optimizer = torch.optim.SGD(params, weight_decay=wd, lr=lr, momentum=0.8)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 50], gamma=0.1)
        tr_dict[name] = {
            "model_state_dict": [model.state_dict(), 0],
            "learnable_params": sum(torch.numel(p) for p in model.parameters()),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch_done": 0,
            "training_loss": [],
            "validation_loss": [],
            "validation_acc": []
        }
    torch.save(tr_dict, base_path + "saved_models/exercise4_FashionMNIST.pth")

tr_dict = torch.load(base_path + "saved_models/exercise4_FashionMNIST.pth")

for model_type in model_list:
    if not brevitas_model:
        model, name = model_type[0](model_type[3], model_type[4], model_type[5], model_type[6]), model_type[1]
    else:
        model, name = model_type[0](), model_type[1]
    lr = model_type[2]
    wd = L2_lambda/lr
    print(name)
    if tr_dict[name]["epoch_done"] < epochs:
        print("training model: ", name)
        model.load_state_dict(tr_dict[name]["model_state_dict"][0])
        model.to(device)
        params = model.parameters()
        optimizer = torch.optim.SGD(params, weight_decay=wd, lr=lr, momentum=0.8)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 50], gamma=0.1)
        optimizer.load_state_dict(tr_dict[name]["optimizer_state_dict"])
        scheduler.load_state_dict(tr_dict[name]["scheduler_state_dict"])
        if tr_dict[name]["epoch_done"] == 0:
            best_acc = 0
        else:
            best_acc = tr_dict[name]["model_state_dict"][1]

        for t in tqdm(range(epochs - tr_dict[name]["epoch_done"])):
            print(f"Epoch {t+1}\n-------------------------------")
            loss = train(train_dataloader, model, loss_fn, optimizer, tr_dict[name]["training_loss"])
            current_acc = test(validation_dataloader, model, loss_fn, tr_dict[name]["validation_loss"])
            scheduler.step()
            tr_dict[name]["validation_acc"].append(current_acc * 100)
            tr_dict[name]["epoch_done"] += 1
            tr_dict[name]["optimizer_state_dict"] = optimizer.state_dict()
            tr_dict[name]["scheduler_state_dict"] = scheduler.state_dict()
            if current_acc > best_acc:
                best_acc = current_acc
                tr_dict[name]["model_state_dict"] = [model.state_dict(), best_acc]
            torch.save(tr_dict, base_path + "saved_models/exercise4_FashionMNIST.pth")
            print(f"lr: {optimizer.param_groups[0]['lr']:0.2e}\n")

###-----

import numpy as np
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

tr_dict = torch.load(base_path + "saved_models/exercise4_FashionMNIST.pth")

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']

Path(base_path + "ex4_figures_FM").mkdir(parents=True, exist_ok=True)

for model_type in model_list:
    if not brevitas_model:
        model, name = model_type[0](model_type[3], model_type[4], model_type[5], model_type[6]), model_type[1]
    else:
        model, name = model_type[0](), model_type[1]
    print(name, ":")
    print("\tLearnable parameters: {:d}".format(tr_dict[name]["learnable_params"]))
    print("\tValidation Accuracy: {:.1f} %".format(tr_dict[name]["model_state_dict"][1] * 100))
    model.load_state_dict(tr_dict[name]["model_state_dict"][0])
    model.to(device)

    #training loss
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
    plt.ylim(top=3, bottom=0)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training loss", "Validation loss"])
    plt.savefig(base_path + f"ex4_figures_FM/{name}_loss_training_loss.png")
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
    plt.savefig(base_path + f"ex4_figures_FM/{name}_loss_validation_accuracy.png")
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
    plt.savefig(base_path + f"ex4_figures_FM/{name}_loss_single_class_accuracy.png")
    plt.clf()

    print("\tMin per-class accuracy is %.4f for class %s\n" %(min_correct[1], min_correct[0]))

    ##----

if not brevitas_model:
    model_list_2 = [(quant_custom_mini_resnet, 1e-4, 8, 8, 16, "scale"), (quant_custom_mini_resnet, 1e-4, 16, 16, 32, "scale")]
    #model_list_2 = [(quant_custom_mini_resnet, 1e-4, 4, 4, 8, "scale")]
    index = 0

    for model_type in model_list:
        name = model_type[1]
        print(name, "w/o BN :")
        folded_model = model_list_2[index][0](model_list_2[index][2], model_list_2[index][3], model_list_2[index][4], model_list_2[index][5])
        folded_state_dict = torch.load(base_path + "saved_models/exercise4_FashionMNIST.pth")[name]["model_state_dict"][0]

        ### fold and quantize the weights, remove batch norm parameters
        keys = list(folded_state_dict.keys())
        for key in keys:
            m = re.match(r"conv2D_(\d+)\.weight", key)
            if m is not None:
                nbr = m.groups()[0]
                
                min_v_w = folded_model.min_v_w
                max_v_w = folded_model.max_v_w
                min_v_b = folded_model.min_v_b
                max_v_b = folded_model.max_v_b

                quant_method = folded_model.quant_method
                weight = folded_state_dict[key]
                ws = weight.shape[0]
                bias = folded_state_dict["conv2D_" + nbr + ".bias"]
                bn_gamma = folded_state_dict["conv2D_" + nbr + ".batch_norm.weight"]
                bn_beta = folded_state_dict["conv2D_" + nbr + ".batch_norm.bias"]
                bn_mean = folded_state_dict["conv2D_" + nbr + ".batch_norm.running_mean"]
                bn_var = folded_state_dict["conv2D_" + nbr + ".batch_norm.running_var"]
                bn_eps = 1e-5
                new_weight = weight * bn_gamma.reshape((ws,1,1,1)) / torch.sqrt(bn_var.reshape((ws,1,1,1)) + bn_eps)
                folded_state_dict[key] = myDoReFaWeightQuantization(new_weight, min_v_w, max_v_w, quant_method)
                new_bias = ((bias - bn_mean) * bn_gamma / torch.sqrt(bn_var + bn_eps)) + bn_beta
                folded_state_dict["conv2D_" + nbr + ".bias"] = myDoReFaWeightQuantization(new_bias, min_v_b, max_v_b, quant_method)
                
                #remove useless entries
                folded_state_dict.pop("conv2D_" + nbr + ".batch_norm.weight")
                folded_state_dict.pop("conv2D_" + nbr + ".batch_norm.bias")
                folded_state_dict.pop("conv2D_" + nbr + ".batch_norm.running_mean")
                folded_state_dict.pop("conv2D_" + nbr + ".batch_norm.running_var")
                folded_state_dict.pop("conv2D_" + nbr + ".batch_norm.num_batches_tracked")
        
        folded_model.load_state_dict(folded_state_dict)
        folded_model.to(device)

        #test accuracy
        accuracy = test(test_dataloader, folded_model, loss_fn, verbose=False)
        print("\tTest Accuracy: {:.1f} %".format(accuracy * 100))

        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        with torch.no_grad():
            for X, y in test_dataloader:
                images, labels = X.to(device), y.to(device)
                outputs = folded_model(images)
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
        plt.savefig(base_path + f"ex4_figures_FM/{name}_without_BN_single_class_accuracy.png")
        plt.clf()

        print("\tMin per-class accuracy is %.4f for class %s\n" %(min_correct[1], min_correct[0]))

"""
In this last exercise you will port the CNN exported with Brevitas to FINN on custom accelerator implemented on a
ZCU104 development board. Due to limitations on the FINN version made available to the public by Xilinx and time
constraint of the special project, you will only port the fashion MNIST model to FPGA.

FINN uses PYNQ to deploy the bitstream to the FPGA, you can read more information on PYNQ here:
https://pynq.readthedocs.io/en/latest/index.html
More information on our specific development board can be found here.
https://pynq.readthedocs.io/en/latest/getting_started/zcu104_setup.html

For this exercise you should have already implemented, trained and exported the CNN model in FINN format.
The Vivado 2020.1 suite, FINN docker and ZCU104 development board that are required to complete this task have been
prepared and tested previously. You should use the remote server during all the design and deployment phases of this
exercise. Check the telegram channel for login instructions, each student has its own home directory and credentials.
The IP name of the pynq board is pynq-zcu104, IP address is 192.168.166.58. You can connect to this IP only while using
the university WiFi/Ethernet. The same thing applies to the server with all the tools and necessary computing resources.
Therefore, you can only complete this exercise while being at the university.
Since you will not have physical access to the server, you will need to connect with the following command from a linux
machine ssh -X <your username>@pc-lse-1861.polito.it in order to use GUI applications.
Otherwise, if you are on windows, you should use X2GO or Mobaxterm. With Mobaxterm you should be able to work also when
you are not on university ground, as it supports ssh tunnelling.

Assignment:
- Read the FINN documentation before writing any code or executing the notebooks. If any problem occurs during the
  execution of the cells inside the jupyter notebooks, it is probably because of some wrong configuration done with the
  environment variables. https://finn.readthedocs.io/en/latest/getting_started.html#quickstart
- Read and understand the FINN tutorials, you can launch them using the command "./run-docker.sh notebook" inside the
  FINN folder in /home/tools/FINN .
  A summary of what is done in the tutorials can be found here: https://finn.readthedocs.io/en/latest/tutorials.html
- For any problems, first check the FAQ https://finn.readthedocs.io/en/latest/faq.html, then the official GitHub
  repository discussion page https://github.com/Xilinx/finn/discussions. For Brevitas issue please refer to its own
  gitter https://gitter.im/xilinx-brevitas/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
- After you have completed all tutorials, follow the instructions of /end2end_example/cybersecurity tutorials to deploy,
  validate and extract hardware metrics of your CNN model running on our ZCU104 development board.
  To complete this exercise you have to provide the hardware metrics of your model as presented at the end of the last
  tutorial, i.e., end2end_example/cybersecurity/3-build-accelerator-with-finn.ipynb
- In the final report you will have to include the code of the jupiter notebooks that you modified to deploy the CNN.
"""