from pickletools import optimize
import torch, torchvision, copy
from torch import nn, true_divide
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
from exercise_4 import quant_custom_mini_resnet, quant_brevitas_mini_resnet, PACT_QuantReLU
import quantization as cs
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import brevitas.config

brevitas.config.IGNORE_MISSING_KEYS=True


Path("./runs/exercise_4_FashionMNIST").mkdir(parents=True, exist_ok=True)  # check if runs directory for tensorboard exist, if not create one

writer = SummaryWriter('runs/exercise_FashionMNIST')

transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2862], std=[0.3204]), transforms.RandomHorizontalFlip()])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2862], std=[0.3204])])

training_data, validation_data = random_split(datasets.FashionMNIST(root="data", train=True, download=True, transform=transform_train), [50000, 10000])
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform_test)

best_workers = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
device = "cpu"

model = quant_custom_mini_resnet()
prova = model.quant_method
model_parameters = filter(lambda p: p.requires_grad, model.parameters())

params = sum([np.prod(p.size()) for p in model_parameters])
memory = params * 32 / 8 / 1024 / 1024
print("this model has ", params, " parameters")
print("total weight memory is %.4f MB" %(memory))

#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.,0.5,1.1,0.55,1.,0.5,2.5,0.5,0.5,0.5]).cuda())

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
        #alpha_opt.zero_grad()

        loss.backward()

        optimizer.step()
        #alpha_opt.step()

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

batch_size = [32]
lr = 1e-3
epochs = 10
best_correct = 0
Train = False
Path("./saved_models").mkdir(parents=True, exist_ok=True)
print("Use $ tensorboard --logdir=runs/exercise_FashionMNIST to access training statistics")
if Train == True:
    for batch in batch_size:
        model = quant_custom_mini_resnet()
        model.to(device)

        train_dataloader = DataLoader(training_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())
        validation_dataloader = DataLoader(validation_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())
        test_dataloader = DataLoader(test_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())
        
        params = model.parameters()
        '''
        alpha_par = list()
        for mod in model.modules():
            
            if isinstance(mod, cs.ReLu):
                alpha_par += list(mod.parameters())
            else:
                params += list(mod.parameters())

            if isinstance(mod, PACT_QuantReLU):
                alpha_par += list(mod.parameters())
            else:
                params += list(mod.parameters())
            
        '''
        ### should alpha have its own weight_decay ??
        '''
        optimizer = torch.optim.SGD(params, weight_decay=0.001, lr=lr)
        alpha_opt = torch.optim.SGD(alpha_par, weight_decay=0.01, lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6,8], gamma=0.2, verbose=True)
        alpha_sched = torch.optim.lr_scheduler.MultiStepLR(alpha_opt, milestones=[6,8], gamma=0.2, verbose=True)
        '''
        optimizer = torch.optim.Adam(params, lr, amsgrad=False, weight_decay=1e-10)

        print(f"using: batch={batch}, n_ep={epochs}, lr={lr}")
        for t in tqdm(range(epochs)):
            print(f"Epoch {t+1}\n-------------------------------")
            loss = train(train_dataloader, model, loss_fn, optimizer, t)
            current_correct = test(validation_dataloader, model, loss_fn)
            '''
            scheduler.step()
            alpha_sched.step()
            '''
            writer.add_scalar('test accuracy', current_correct, t)
            writer.flush()
            if current_correct > best_correct:
                best_correct = current_correct
                torch.save({
                    "batch_size": batch,
                    "lr": lr,
                    'epoch': t,
                    'model_state_dict': model.state_dict(),
                    'loss': loss,
                    'test_acc': current_correct,
                }, "./saved_models/exercise4FashionMNIST.pth")
        #break

writer.close()
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())

###switch to brevitas
model = quant_brevitas_mini_resnet()
brevitas_state_dict = torch.load("./saved_models/brevitas.pth")["model_state_dict"]
min_v = model.min_v_w
max_v = model.max_v_w
quant_method = model.quant_method
###load the best model for brevitas check
opt_model = torch.load("./saved_models/exercise4FashionMNIST.pth")
print(opt_model["test_acc"])
opt_state_dict = opt_model["model_state_dict"]
###quantize parameters back
for tensor in opt_state_dict:
    opt_state_dict[tensor] = cs.myDoReFaWeightQuantization.quantize(opt_model["model_state_dict"][tensor], min_v, max_v, quant_method)
'''
opt_state_dict["identity.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"] = brevitas_state_dict["identity.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"]
opt_state_dict["act_1.relu.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"] = brevitas_state_dict["act_1.relu.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"]
opt_state_dict["act_2.relu.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"] = brevitas_state_dict["act_2.relu.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"]
opt_state_dict["act_3.relu.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"] = brevitas_state_dict["act_3.relu.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"]
opt_state_dict["act_4.relu.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"] = brevitas_state_dict["act_4.relu.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"]
opt_state_dict["act_5.relu.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"] = brevitas_state_dict["act_5.relu.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"]
'''

###load the dictionary
model.load_state_dict(opt_state_dict)

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