from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import cohen_kappa_score,matthews_corrcoef
from sklearn.metrics import average_precision_score

plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'train_test_IPdataset'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#device="cuda"
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

        
def test(model):
    pred = []
    true = []
    pred_wrong = []
    true_wrong = []
    image = []
    running_corrects = 0
    running_loss=0
    sm = nn.Softmax(dim = 1)
    for batch_idx,(inputs, labels) in enumerate(dataloaders['val']):
        data= inputs.to(device)
        target=labels.to(device)
        #data = data.type(device.FloatTensor)
        #target = target.type(device.LongTensor)
        model.eval()
        output = model(data)
        output = sm(output)
        _, preds = torch.max(output, 1)
        #running_corrects = running_corrects + torch.sum(preds == target.data)
        #running_loss += loss.item() * data.size(0)
        preds = preds.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        preds = np.reshape(preds,(len(preds),1))
        target = np.reshape(target,(len(preds),1))
        data = data.cpu().detach().numpy()

        for i in range(inputs.size()[0]):
            pred.append(preds[i])
            true.append(target[i])
            if(preds[i]!=target[i]):
                pred_wrong.append(preds[i])
                true_wrong.append(target[i])
                image.append(data[i])
                #epoch_acc = running_corrects.double()/(len(dataloader)*batch_size)
                #epoch_loss = running_loss/(len(dataloader)*batch_size)
                #print(epoch_acc,epoch_loss)

	return true,pred,image,true_wrong,pred_wrong
    
    
def wrong_plot(true,ima,pred,n_figures = 8):
    print(len(true))
    print(len(ima))
    print(len(pred))
    classes = image_datasets['train'].classes
    #encoder and decoder to convert classes into integer
    decoder = {}
    for i in range(len(classes)):
        decoder[classes[i]] = i
    encoder = {}
    for i in range(len(classes)):
        encoder[i] = classes[i]
        
    inv_normalize =  transforms.Normalize(
    mean=[-0.4302/0.2361, -0.4575/0.2347, -0.4539/0.2432],
    std=[1/0.2361, 1/0.2347, 1/0.2432])
    
    print('Classes in order Actual and Predicted')
    n_row = int(n_figures/4)
    fig,axes = plt.subplots(figsize=(10,5), nrows = n_row, ncols=4)
    for ax in axes.flatten():
        a = random.randint(0,len(true)-1)
    
        image,correct,wrong = ima[a],true[a],pred[a]
        image = torch.from_numpy(image)
        correct = int(correct)
        c = encoder[correct]
        wrong = int(wrong)
        w = encoder[wrong]
        f = 'A:'+c + ',' +'P:'+w
        if inv_normalize !=None:
            image = inv_normalize(image)
        image = image.numpy().transpose(1,2,0)
        im = ax.imshow(image)
        ax.set_title(f)
        ax.axis('off')
    plt.savefig('wrong_predictions.jpg')
    return plt

def performance_matrix(true,pred):
    print("true len",len(true))
    print("pred len",len(pred))
    precision = average_precision_score(true,pred,average='macro')
    #recall = metrics.average_recall_score(true,pred,average='macro')
    accuracy = accuracy_score(true,pred)
    f1 = f1_score(true,pred,average='macro')
    mcc= matthews_corrcoef(true,pred)
    kappa= cohen_kappa_score(true,pred)
    
    #print('Confusion Matrix:\n',metrics.confusion_matrix(true, pred))
    print('Precision: {}, Accuracy: {}: ,f1_score: {}, MCC: {},Kappa Score: {}'.format(precision,accuracy,f1,mcc,kappa))
    
    
 
    
    


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft=model_ft.to(device)

#model_ft = models.googlenet(pretrained=True)
# set_parameter_requires_grad(model_ft, feature_extract)
#num_ftrs = model_ft.classifier[6].in_features
#model_ft.classifier[6] = nn.Linear(num_ftrs,2)
# input_size = 224 model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)


true,pred,image,true_wrong,pred_wrong=test(model_ft)
plt_obj=wrong_plot(true_wrong,image,pred_wrong)
plt_obj.savefig('wrong_predictions2.png')
performance_matrix(true,pred)
