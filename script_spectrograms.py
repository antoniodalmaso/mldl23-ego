import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim
import torch
from utils.loaders import ActioNetDataset
import sys
import numpy as np
from torch.utils.data import random_split
from models.ActioNet.SpecNet import SpecNet
from sklearn.model_selection import KFold

np.random.seed(13696641)
torch.manual_seed(13696641)

def main():
    _, path_emg, path_rgb, learning_rate, momentum, epochs = sys.argv
    learning_rate = float(learning_rate)
    momentum = float(momentum)
    epochs = int(epochs)
    
    # DATASETS #
    dataset = ActioNetDataset(base_data_path=path_emg, rgb_path=path_rgb, num_clips=1, modality="EMG")
    
    ##########################################################################################################
    splits = KFold(5, shuffle=True, random_state=13696641)
    accuracies = []
    for fold, (train_indx, val_indx) in enumerate(splits.split(dataset)):
      model = SpecNet()
      model.load_state_dict(torch.load("/content/mldl23-ego/pretrained_specnet/specnet_weights_no_SE.pt"))

      print(f'FOLD: {fold}')
      data_train = torch.utils.data.Subset(dataset, train_indx)
      data_val = torch.utils.data.Subset(dataset, val_indx)
      
      trainloader = torch.utils.data.DataLoader(data_train, batch_size=16, shuffle=True, num_workers=2)
      testloader = torch.utils.data.DataLoader(data_val, batch_size=16, shuffle=True, num_workers=2)
    ##########################################################################################################

    # NETWORK #

      loss_function = nn.CrossEntropyLoss()
      optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

      # TRAIN #
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      model = model.to(device)
      train(model=model, trainloader=trainloader, optimizer=optimizer, loss_function=loss_function, device=device, epochs=epochs)

      # VALIDATE #
      accuracies.append(validate(model=model, testloader=testloader, device=device))
    print(f"Accuracies on the single splits: {np.array(accuracies)*100}")
    print(f"Average accuracy: {np.mean(accuracies) * 100}")

def train(model, trainloader, optimizer, loss_function, device, epochs):
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_correct = 0
        running_total = 0

        for inputs, labels in trainloader:
            inputs["EMG"] = inputs["EMG"].to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs["EMG"])
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # computing accuracy
            _, predicted = torch.max(outputs, -1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)

        print(f'[epoch {epoch + 1}] accuracy: {100 * running_correct / running_total:.3f}%')
    print('Finished Training!')

def validate(model, testloader, device):
    model.eval()
    class_correct = list(0. for i in range(21))
    class_total = list(0. for i in range(21))

    with torch.no_grad():
        for (inputs, labels) in testloader:
            inputs["EMG"] = inputs["EMG"].to(device)
            labels = labels.to(device)
            
            outputs = model(inputs["EMG"])
            _, predicted = torch.max(outputs, -1)
            
            c = (predicted == labels).squeeze()
            for i in range(labels.shape[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
    for i in range(21):
        print('Accuracy of %5s : %2d %%' % (
            i, 100 * class_correct[i] / class_total[i]))
    print(f"Total Acc: {np.sum(class_correct) / np.sum(class_total)*100:.3f}")
    return np.sum(class_correct) / np.sum(class_total)


if __name__ == '__main__':
    main()
