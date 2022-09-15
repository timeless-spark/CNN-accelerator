import torch, torchvision, copy
from torch import nn, true_divide
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
from exercise_4 import FINN_quant_brevitas_mini_resnet
from pathlib import Path
import brevitas.config
from brevitas.export.onnx.finn.manager import FINNManager
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
import brevitas.nn as qnn
from brevitas.quant_tensor import QuantTensor 

brevitas.config.IGNORE_MISSING_KEYS=True

transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2862], std=[0.3204]), transforms.RandomHorizontalFlip()])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2862], std=[0.3204])])

training_data, validation_data = random_split(datasets.FashionMNIST(root="data", train=True, download=True, transform=transform_train), [50000, 10000])
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform_test)

FINN_save = True
best_workers = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.,0.5,1.1,0.55,1.,0.5,2.5,0.5,0.5,0.5]).to(device))

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
            correct += (pred.value.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    if loss_list is not None:
        loss_list.append(test_loss)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

batch = 24
lr = 1e-2
epochs = 15
best_correct = 0
Train = True

Path("./saved_models").mkdir(parents=True, exist_ok=True)

res_dict = {
    "FINN_resnet": []
}

train_dataloader = DataLoader(training_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=False)
validation_dataloader = DataLoader(validation_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=False)
test_dataloader = DataLoader(test_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=False)

model = FINN_quant_brevitas_mini_resnet()
model.to(device)

if Train == True:
        model = FINN_quant_brevitas_mini_resnet()
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        memory = params * 32 / 8 / 1024 / 1024

        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), weight_decay=lr/128, momentum=.8, lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,4], gamma=0.2, verbose=True)
        best_correct = 0
        training_loss = []
        avg_loss = []

        for t in tqdm(range(epochs)):
            print(f"Epoch {t+1}\n-------------------------------")
            loss = train(train_dataloader, model, loss_fn, optimizer, training_loss, batch)
            current_correct = test(test_dataloader, model, loss_fn, avg_loss)
            scheduler.step()
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
                }, "./saved_models/exercise5.pth")

classes = test_data.classes

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

###load the best model..
opt_model = torch.load("./saved_models/exercise5.pth")
print(opt_model["test_acc"])
model.load_state_dict(opt_model["model_state_dict"])
model.to(device)

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

res_dict["FINN_resnet"].append([batch, lr, opt_model["test_acc"], opt_model["loss"], lowest_class_accuracy, training_loss, avg_loss])

Path("./saved_models").mkdir(parents=True, exist_ok=True)

torch.save(res_dict, "./saved_models/ex5_res.pth")

Path("./FINN_export").mkdir(parents=True, exist_ok=True)

if FINN_save:
    #if dataset_type == 'FashionMNIST':
    in_tensor = (1, 1, 28, 28)
    input_qt = np.random.randint(0, 255, in_tensor).astype(np.float32)
    #elif dataset_type == 'CIFAR10':
    #    in_tensor = (1, 3, 32, 32)
    #else:
    #    exit("invalid dataset")
    FINNManager.export(model.to("cpu"), export_path="./FINN_export/" + "FashionMNISTfinn.onnx", input_shape=in_tensor)#, input_t = QuantTensor(torch.from_numpy(input_qt), signed = False, scale=torch.tensor(1.0), bit_width=torch.tensor(8.0)))
    BrevitasONNXManager.export(model.cpu(), export_path="./FINN_export/" + "FashionMNISTbrevitas.onnx", input_shape=in_tensor)#, input_t = QuantTensor(torch.from_numpy(input_qt), signed = False, scale=torch.tensor(1.0), bit_width=torch.tensor(8.0)))
    print("Succesfully written FINN export files!")


