from .SpecNet import SpecNet
import torch.nn as nn
import torch

class MultiNet(nn.Module):
    def __init__(self):
        super(MultiNet, self).__init__()
        
        self.specnet = SpecNet(predict = False)
        self.lstm_emg = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.lstm_rgb = nn.LSTM(input_size=1024, hidden_size=1024, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 + 1024, 512)
        self.drop = nn.Dropout(p=0.15)
        self.hswish = nn.Hardswish()
        self.fc2 = nn.Linear(512, 21)
        
    def forward(self, x):
        x_rgb = x["RGB"]
        x_emg = x["EMG"]
        
        # RGB features
        x_rgb, _ = self.lstm_rgb(x_rgb)
        x_rgb = x_rgb[:,-1,:]
        
        # EMG features
        x_emg = self.specnet(x_emg)
        x_emg, _ = self.lstm_emg(x_emg)
        x_emg = x_emg[:,-1,:]
        
        # concatenation
        x = torch.hstack((x_rgb, x_emg))
        
        # fully connected
        x = self.fc1(x)
        x = self.drop(x)
        x = self.hswish(x)
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=0)
        
        return x