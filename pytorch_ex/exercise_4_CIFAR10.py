from posixpath import split
from tabnanny import verbose
import torch, torchvision, copy
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import transforms
import brevitas
import numpy as np
from tqdm import tqdm
from exercise_4 import quant_custom_ex3ResNet_medium, quant_custom_ex3ResNet_medium_folded, quant_brevitas_ex3ResNet_medium
import quantization as cs
from pathlib import Path
import re

brevitas.config.IGNORE_MISSING_KEYS=True

base_path = "./"

Path(base_path + "saved_models").mkdir(parents=True, exist_ok=True)

initialize_dict = False
train_dict = True

#scheduler_step = False

transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), transforms.RandomResizedCrop(size=(32,32), scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(0.5)])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
training_data, validation_data = random_split(datasets.CIFAR10(root="data", train=True, download=True, transform=transform_train), [45000, 5000])
test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)

#best_workers = find_num_workers(training_data=training_data, batch_size=batch_size)
best_workers = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
#device = "cpu"

#model_list = [(quant_custom_ex3ResNet_medium_folded, "quant_custom_ex3ResNet_medium_4"), (quant_custom_ex3ResNet_medium_folded, "quant_custom_ex3ResNet_medium_5_5", 5e-5), (quant_custom_ex3ResNet_medium_folded, "quant_custom_ex3ResNet_medium_5", 1e-5)]
#touple format (model, "model_name", lr, weight_bit, act_bit, bias_bit, quant_method)
#model_list = [(quant_custom_ex3ResNet_medium_folded, "quant_custom_ex3ResNet_medium_4", 8e-4, 2e-1, 4, 8, 8, "scale"), (quant_custom_ex3ResNet_medium_folded, "quant_custom_ex3ResNet_medium_8", 5e-3, 2e-1, 8, 8, 16, "scale"), (quant_custom_ex3ResNet_medium_folded, "quant_custom_ex3ResNet_medium_16", 8e-3, 2e-1, 16, 16, 32,"scale")]
model_list = [(quant_custom_ex3ResNet_medium_folded, "quant_custom_ex3ResNet_medium_4", 8e-4, 4, 8, 8, "scale")]
#model_list = [(quant_custom_ex3ResNet_medium_folded, "quant_custom_ex3ResNet_medium_8", 5e-3, 8, 8, 16, "scale")]
#model_list = [(quant_custom_ex3ResNet_medium_folded, "quant_custom_ex3ResNet_medium_16", 5e-3, 16, 16, 32,"scale")]
#model_list = [(quant_brevitas_ex3ResNet_medium, "quant_brevitas_ex3ResNet_medium", 1e-3)]
brevitas_model = False

for p in model_list:
    print(p[1])

def return_model_params(model, train_alpha=True):
    if train_alpha:
        params = model.parameters()
    else:
        params = list()
        for mod in model.modules():
            if not isinstance(mod, cs.ReLu):
                params += list(mod.parameters())
    return params

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

        if batch % 60 == 0:
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

batch_size = 128
L2_lambda = 1e-9
pre_epochs = 25
SGD_epochs = 50
Adam_epochs = 150
loss_fn = nn.CrossEntropyLoss()

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())

print("train dataset samples: ", len(train_dataloader.dataset))
print("validation dataset samples: ", len(validation_dataloader.dataset))
print("test dataset samples: ", len(test_dataloader.dataset))

if initialize_dict:
    tr_dict = {}

    # model initialization, floating point training..
    if not brevitas_model and train_dict:
        model = model_list[0][0](model_list[0][4], model_list[0][5], model_list[0][6], model_list[0][7], False)
        lr = 1e-2
        model.to(device)
        params = return_model_params(model, train_alpha=False)
        optimizer = torch.optim.SGD(params, weight_decay=L2_lambda/lr, lr=lr, momentum=0.8)
        for t in tqdm(range(pre_epochs)):
            print(f"Epoch {t+1}\n-------------------------------")
            loss = train(train_dataloader, model, loss_fn, optimizer)
            test(validation_dataloader, model, loss_fn)

    for model_type in model_list:
        if brevitas_model:
            model = model_type[0]()
        elif not train_dict:
            model = model_type[0](model_type[4], model_type[5], model_type[6], model_type[7], True)
        name = model_type[1]
        params = return_model_params(model)
        tr_dict[name] = {
            "model_state_dict": [model.state_dict(), 0],
            "learnable_params": sum(torch.numel(p) for p in return_model_params(model)),
            "epoch_done": 0,
            "training_loss": [],
            "validation_loss": [],
            "validation_acc": []
        }
    torch.save(tr_dict, base_path + "saved_models/exercise4_CIFAR10.pth")

tr_dict = torch.load(base_path + "saved_models/exercise4_CIFAR10.pth")

for model_type in model_list:
    if not brevitas_model:
        model, name = model_type[0](model_type[4], model_type[5], model_type[6], model_type[7], True), model_type[1]
    else:
        model, name = model_type[0](), model_type[1]
    lr = model_type[2]
    lr_decay = model_type[3]
    wd = L2_lambda/lr
    print(name)
    if tr_dict[name]["epoch_done"] < SGD_epochs:
        print("training model: ", name)
        model.load_state_dict(tr_dict[name]["model_state_dict"][0])
        model.to(device)
        params = return_model_params(model)
        optimizer = torch.optim.SGD(params, weight_decay=wd, lr=lr, momentum=0.8)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=lr_decay, total_iters=SGD_epochs, verbose=True)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 90], gamma=0.1)
        if tr_dict[name]["epoch_done"] == 0:
            best_acc = 0
        else:
            best_acc = tr_dict[name]["model_state_dict"][1]
            optimizer.load_state_dict(tr_dict[name]["optimizer_state_dict"])

        for t in tqdm(range(SGD_epochs - tr_dict[name]["epoch_done"])):
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
            torch.save(tr_dict, base_path + "saved_models/exercise4_CIFAR10.pth")
            print(f"lr: {optimizer.param_groups[0]['lr']:0.2e}\n")

if not brevitas_model:
    for model_type in model_list:
        if not brevitas_model:
            model, name = model_type[0](model_type[4], model_type[5], model_type[6], model_type[7], True), model_type[1]
        else:
            model, name = model_type[0](), model_type[1]
        lr = 5e-5
        wd = L2_lambda/lr
        print(name)
        if tr_dict[name]["epoch_done"] < SGD_epochs+Adam_epochs:
            print("training model: ", name)
            model.load_state_dict(tr_dict[name]["model_state_dict"][0])
            model.to(device)
            params = return_model_params(model)
            optimizer = torch.optim.Adam(params, weight_decay=wd, lr=lr, eps=1e-5)
            if tr_dict[name]["epoch_done"] == SGD_epochs:
                best_acc = 0
            else:
                best_acc = tr_dict[name]["model_state_dict"][1]
                optimizer.load_state_dict(tr_dict[name]["optimizer_state_dict"])

            for t in tqdm(range(SGD_epochs+Adam_epochs - tr_dict[name]["epoch_done"])):
                print(f"Epoch {t+1}\n-------------------------------")
                loss = train(train_dataloader, model, loss_fn, optimizer, tr_dict[name]["training_loss"])
                current_acc = test(validation_dataloader, model, loss_fn, tr_dict[name]["validation_loss"])
                tr_dict[name]["validation_acc"].append(current_acc * 100)
                tr_dict[name]["epoch_done"] += 1
                tr_dict[name]["optimizer_state_dict"] = optimizer.state_dict()
                if current_acc > best_acc:
                    best_acc = current_acc
                    tr_dict[name]["model_state_dict"] = [model.state_dict(), best_acc]
                torch.save(tr_dict, base_path + "saved_models/exercise4_CIFAR10.pth")
                print(f"lr: {optimizer.param_groups[0]['lr']:0.2e}\n")

torch.save(tr_dict, base_path + "saved_models/exercise4_CIFAR10.pth")

###-----

import numpy as np
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

tr_dict = torch.load(base_path + "saved_models/exercise4_CIFAR10.pth", map_location=torch.device('cpu'))

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']

Path(base_path + "ex4_figures_CIFAR10").mkdir(parents=True, exist_ok=True)

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

for model_type in model_list:
    if not brevitas_model:
        model, name = model_type[0](model_type[3], model_type[4], model_type[5], model_type[6], True), model_type[1]
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
    plt.savefig(base_path + f"ex4_figures_CIFAR10/{name}_loss_training_loss.png")
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
    plt.savefig(base_path + f"ex4_figures_CIFAR10/{name}_loss_validation_accuracy.png")
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
    plt.savefig(base_path + f"ex4_figures_CIFAR10/{name}_loss_single_class_accuracy.png")
    plt.clf()

    print("\tMin per-class accuracy is %.4f for class %s\n" %(min_correct[1], min_correct[0]))

if not brevitas_model:
    #model_list_2 = [(quant_custom_ex3ResNet_medium, 8e-3, 2e-1, 16, 16, 32,"scale")]
    model_list_2 = [(quant_custom_ex3ResNet_medium, 5e-3, 2e-1, 8, 8, 16, "scale")]
    #model_list_2 = [(quant_custom_ex3ResNet_medium, 8e-4, 2e-1, 4, 8, 8, "scale")]
    index = 0

    for model_type in model_list:
        name = model_type[1]
        print(name, "w/o BN :")
        folded_model = model_list_2[index][0](model_list_2[index][3], model_list_2[index][4], model_list_2[index][5], model_list_2[index][6], True)
        folded_state_dict = tr_dict[name]["model_state_dict"][0]

        ### fold and quantize the weights, remove batch norm parameters
        keys = list(folded_state_dict.keys())
        for key in keys:
            splitted = key.split(".")
            l = len(splitted)
            f_key = splitted[l-2] + "." + splitted[l-1]
            check = False

            m = re.match(r"conv1\.weight", f_key)
            if m is not None:
                base = splitted[0]
                check = True
            m = re.match(r"l.\.weight", f_key)
            if m is not None:
                base = splitted[0] + "." + splitted[1] + "." + splitted[2]
                check = True
            m = re.match(r"up\.weight", f_key)
            if m is not None:
                base = splitted[0] + "." + splitted[1]
                check = True

            if check:
                min_v_w = folded_model.min_v_w
                max_v_w = folded_model.max_v_w
                min_v_b = folded_model.min_v_b
                max_v_b = folded_model.max_v_b

                quant_method = folded_model.quant_method
                weight = folded_state_dict[base + ".weight"]
                ws = weight.shape[0]
                bias = folded_state_dict[base + ".bias"]
                bn_gamma = folded_state_dict[base + ".batch_norm.weight"]
                bn_beta = folded_state_dict[base + ".batch_norm.bias"]
                bn_mean = folded_state_dict[base + ".batch_norm.running_mean"]
                bn_var = folded_state_dict[base + ".batch_norm.running_var"]
                bn_eps = 1e-3
                new_weight = weight * bn_gamma.reshape((ws,1,1,1)) / torch.sqrt(bn_var.reshape((ws,1,1,1)) + bn_eps)
                folded_state_dict[key] = cs.myDoReFaWeightQuantization(new_weight, min_v_w, max_v_w, quant_method)
                new_bias = ((bias - bn_mean) * bn_gamma / torch.sqrt(bn_var + bn_eps)) + bn_beta
                folded_state_dict[base + ".bias"] = cs.myDoReFaWeightQuantization(new_bias, min_v_b, max_v_b, quant_method)
                
                #remove useless entries
                folded_state_dict.pop(base + ".batch_norm.weight")
                folded_state_dict.pop(base + ".batch_norm.bias")
                folded_state_dict.pop(base + ".batch_norm.running_mean")
                folded_state_dict.pop(base + ".batch_norm.running_var")
                folded_state_dict.pop(base + ".batch_norm.num_batches_tracked")
        
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
        plt.savefig(base_path + f"ex4_figures_CIFAR10/{name}_without_BN_single_class_accuracy.png")
        plt.clf()

        print("\tMin per-class accuracy is %.4f for class %s\n" %(min_correct[1], min_correct[0]))