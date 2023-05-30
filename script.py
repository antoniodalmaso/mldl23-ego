import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim
import torch
from utils.loaders import I3DFeaturesDataset
import sys
import numpy as np
from models.EpicKitchens.EpicKitchensLSTM import  EpicKitchensLSTM
from models.EpicKitchens.EpicKitchensFC import  EpicKitchensFC

np.random.seed(13696641)
torch.manual_seed(13696641)

def main():
    _, path_train, path_test, learning_rate, momentum, epochs = sys.argv
    learning_rate = float(learning_rate)
    momentum = float(momentum)
    epochs = int(epochs)
    
    # DATASETS #
    trainset = I3DFeaturesDataset(path = path_train)
    testset = I3DFeaturesDataset(path = path_test)
    
    # DATA LOADERS #
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

    # NETWORK #
    model = EpicKitchensLSTM() # LSTM classifier
    # model = EpicKitchensFC() # Fully connected classifier

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # TRAIN #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    train(model=model, trainloader=trainloader, optimizer=optimizer, loss_function=loss_function, device=device, epochs=epochs)

    # VALIDATE #
    validate(model=model, testloader=testloader, device=device)

def train(model, trainloader, optimizer, loss_function, device, epochs):
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_correct = 0
        running_total = 0
    
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
        
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
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
    class_correct = list(0. for i in range(8))
    class_total = list(0. for i in range(8))

    with torch.no_grad():
        for (inputs, labels) in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, -1)
            
            c = (predicted == labels).squeeze()
            for i in range(labels.shape[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
    for i in range(8):
        print('Accuracy of %5s : %2d %%' % (
            i, 100 * class_correct[i] / class_total[i]))
    print(f"Total Acc: {np.sum(class_correct) / np.sum(class_total):.3f}")


if __name__ == '__main__':
    main()
