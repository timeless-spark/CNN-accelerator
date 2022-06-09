import torch, torchvision, copy
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
from torch_neural_networks_library import mini_resnet
from pathlib import Path
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

training_data, validation_data = random_split(datasets.FashionMNIST(root="data", train=True, download=True, transform=transform_train), [50000, 10000])
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform_test)

#best_workers = find_num_workers(training_data=training_data, batch_size=batch_size)
best_workers = 2

'''
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
dataiter = iter(copy.deepcopy(test_dataloader))
images, labels = dataiter.next()
img_grid = torchvision.utils.make_grid(images)
writer.add_image(str(batch_size)+'_FashionMNIST_images', img_grid)

'''
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

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

loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([2.,1.,2.2,1.1,2.,1.,5.,1.,1.,1.]).cuda())
#loss_fn = nn.CrossEntropyLoss()

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

batch_size = [16, 32]
lr = 5e-3
epochs = 5
best_correct = 0
Path("./saved_models").mkdir(parents=True, exist_ok=True)
print("Use $ tensorboard --logdir=runs/exercise_2bis to access training statistics")
for batch in batch_size:
    model = mini_resnet()
    model.to(device)

    train_dataloader = DataLoader(training_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())
    validation_dataloader = DataLoader(validation_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())
    test_dataloader = DataLoader(test_data, batch_size=batch, shuffle=True, num_workers=best_workers, pin_memory=torch.cuda.is_available())
    
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.00001, momentum=.8, lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,3], gamma=0.2, verbose=True)

    print(f"using: batch={batch}, n_ep={epochs}, lr={lr}")
    for t in tqdm(range(epochs)):
        print(f"Epoch {t+1}\n-------------------------------")
        loss = train(train_dataloader, model, loss_fn, optimizer, t)
        current_correct = test(validation_dataloader, model, loss_fn)
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