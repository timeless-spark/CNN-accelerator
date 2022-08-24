from pickletools import optimize
import torch, torchvision, copy
from torch import nn, true_divide
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
from exercise_4 import dummyModel, quant_custom_mini_resnet, quant_brevitas_mini_resnet, PACT_QuantReLU
import quantization as cs
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import brevitas.config
from brevitas.export.onnx.finn.manager import FINNManager
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
import brevitas.nn as qnn
from brevitas.quant_tensor import QuantTensor 

brevitas.config.IGNORE_MISSING_KEYS=True

Path("./runs/exercise_4_FashionMNIST").mkdir(parents=True, exist_ok=True)  # check if runs directory for tensorboard exist, if not create one

writer = SummaryWriter('runs/exercise_FashionMNIST')

transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2862], std=[0.3204]), transforms.RandomHorizontalFlip()])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2862], std=[0.3204])])

training_data, validation_data = random_split(datasets.FashionMNIST(root="data", train=True, download=True, transform=transform_train), [50000, 10000])
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform_test)

FINN_save = True
best_workers = 2

device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
#device = "cpu"

model = quant_brevitas_mini_resnet()
#prova = model.quant_method
model_parameters = filter(lambda p: p.requires_grad, model.parameters())

params = sum([np.prod(p.size()) for p in model_parameters])
memory = params * 32 / 8 / 1024 / 1024
print("this model has ", params, " parameters")
print("total weight memory is %.4f MB" %(memory))

#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.,0.5,1.1,0.55,1.,0.5,2.5,0.5,0.5,0.5]))

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
epochs = 1
best_correct = 0
Train = False
Path("./saved_models").mkdir(parents=True, exist_ok=True)
print("Use $ tensorboard --logdir=runs/exercise_FashionMNIST to access training statistics")
if Train == True:
    for batch in batch_size:
        model = quant_brevitas_mini_resnet()
        model.to(device)

        train_dataloader = DataLoader(training_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=False)#torch.cuda.is_available())
        validation_dataloader = DataLoader(validation_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=False)#torch.cuda.is_available())
        test_dataloader = DataLoader(test_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=False)#torch.cuda.is_available())
        
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

model = dummyModel()
#model = dummyModel()

writer.close()
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=best_workers, pin_memory=False)#torch.cuda.is_available())
"""
###switch to brevitas
model = quant_brevitas_mini_resnet()
#brevitas_state_dict = torch.load("./saved_models/brevitas.pth")["model_state_dict"]
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
"""
if FINN_save:
    #if dataset_type == 'FashionMNIST':
    in_tensor = (1, 1, 28, 28)
    input_qt = np.random.randint(0, 255, in_tensor).astype(np.float32)
    #elif dataset_type == 'CIFAR10':
    #    in_tensor = (1, 3, 32, 32)
    #else:
    #    exit("invalid dataset")
FINNManager.export(model.to("cpu"), export_path="./FINN_export_dummy/" + "FashionMNISTfinn.onnx", input_shape=in_tensor)#, input_t = QuantTensor(torch.from_numpy(input_qt), signed = False, scale=torch.tensor(1.0), bit_width=torch.tensor(8.0)))
BrevitasONNXManager.export(model.cpu(), export_path="./FINN_export_dummy/" + "FashionMNISTbrevitas.onnx", input_shape=in_tensor)#, input_t = QuantTensor(torch.from_numpy(input_qt), signed = False, scale=torch.tensor(1.0), bit_width=torch.tensor(8.0)))
print("Succesfully written FINN export files!")


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