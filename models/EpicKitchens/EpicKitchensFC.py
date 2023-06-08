import torch.nn.functional as F
import torch.nn as nn

class EpicKitchensFC(nn.Module):
    def __init__(self, num_classes=8):
        super(EpicKitchensFC, self).__init__()
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024*5, 512)
        self.fc2 = nn.Linear(512, 384)
        self.fc3 = nn.Linear(384, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        fc1 = self.dropout(self.relu(self.fc1(x)))
        fc2 = self.dropout(self.relu(self.fc2(fc1)))
        fc3 = self.dropout(self.relu(self.fc3(fc2)))
        fc4 = self.fc4(fc3)
        
        pred = F.log_softmax(fc4, dim=-1)
        return pred