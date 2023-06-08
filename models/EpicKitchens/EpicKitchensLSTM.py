import torch.nn.functional as F
import torch.nn as nn
import torch

class EpicKitchensLSTM(torch.nn.Module):
    def __init__(self, input_dim = 1024, hidden_dim = 512, num_clips = 5, num_classes = 8):
        super(EpicKitchensLSTM, self).__init__()
        self.num_clips = num_clips
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim, 384)
        self.fc2 = nn.Linear(384, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # LSTM
        out, _ = self.lstm(x)
        out = out[:,-1,:] # prendo solo l'ultimo output di ogni sequenza
        
        # Fully Connected
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.dropout(self.relu(self.fc2(out)))
        out = self.fc3(out)
        
        # Soft Max
        out = F.log_softmax(out, dim = -1)
        #out = torch.sum(out, dim = 1) # questo serve per "regolarizzare"? (consiglio signor peirone)
        return out