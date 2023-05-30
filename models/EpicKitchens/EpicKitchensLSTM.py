import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch

class EpicKitchensLSTM(torch.nn.Module):
    def __init__(self, input_dim = 1024, hidden_dim = 512, num_clips = 5, num_classes = 8):
        super(EpicKitchensLSTM, self).__init__()
        self.num_clips = num_clips
        self.num_classes = num_classes
        
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, self.num_classes)

#    def forward(self, input):
#        output = []
#        for x in input:
#            # LSTM
#            out, hidden = self.lstm(x[0].view(1, 1, -1))
#            out, hidden = self.lstm(x[1].view(1, 1, -1), hidden)
#            out, hidden = self.lstm(x[2].view(1, 1, -1), hidden)
#            out, hidden = self.lstm(x[3].view(1, 1, -1), hidden)
#            out, _ = self.lstm(x[4].view(1, 1, -1), hidden)
#
#            # Fully Connected
#            out = self.linear(out)
#
#            # Soft Max
#            #out = F.log_softmax(out, dim = -1)
#            output.append(F.log_softmax(out, dim = -1))
#        return torch.vstack(output).squeeze()
    
    def forward(self, x):
        # LSTM
        out, _ = self.lstm(x)
        #out = out[:,-1,:] # prendo solo l'ultimo output di ogni sequenza
        
        # Fully Connected
        out = self.linear(out)
        
        # Soft Max
        out = F.log_softmax(out, dim = -1)
        out = torch.sum(out, dim = 1) # questo serve per "regolarizzare"? (consiglio signor peirone)
        return out