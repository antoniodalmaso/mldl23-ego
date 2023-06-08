from torchvision.ops import SqueezeExcitation
import torch.nn as nn

class SpecNet(nn.Module):
    def __init__(self, fc1_size = 704, predict = True):
        super(SpecNet, self).__init__()
        self.predict = predict # establishes where to stop with the forward step
        
        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.hswish1 = nn.Hardswish()
        self.se1 = SqueezeExcitation(16,16, scale_activation=nn.Hardswish)
        self.se2 = SqueezeExcitation(16,16, scale_activation=nn.Hardswish)
        self.se3 = SqueezeExcitation(16,16, scale_activation=nn.Hardswish)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.hswish2 = nn.Hardswish()
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc1_size, 512)
        self.drop = nn.Dropout(p=0.15)
        self.hswish3 = nn.Hardswish()
        self.fc2 = nn.Linear(512, 21)

    def forward(self, x):
        shape = x.shape
        x = x.view(-1, shape[-3], shape[-2], shape[-1])
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.hswish1(x)
        x1 = self.se1(x)
        x = x + x1
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.hswish2(x)
        x2 = self.se2(x) # ?
        x = x + x2
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.hswish2(x)
        x3 = self.se3(x)
        x = x + x3

        x = self.conv4(x)
        x = self.bn2(x)
        x = self.hswish2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1) # Flatten the tensor
        
        if self.predict:
            x = self.fc1(x)
            x = self.drop(x)
            x = self.hswish3(x)
            x = self.fc2(x)
            x = nn.functional.softmax(x, dim=0)
            
        x = x.view(shape[0], -1, x.shape[-1])
        x = x.squeeze()
        
        return x
    

class SpecNet_2D(nn.Module):
    def __init__(self, fc1_size = 88, predict = True):
        super(SpecNet_2D, self).__init__()
        self.predict = predict # establishes where to stop with the forward step
        
        self.conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.hswish1 = nn.Hardswish()
        #self.se1 = SqueezeExcitation(2,2, scale_activation=nn.Hardswish)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(4)
        self.hswish2 = nn.Hardswish()
        self.conv3 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc1_size, 40)
        self.drop = nn.Dropout(p=0.15)
        self.hswish3 = nn.Hardswish()
        self.fc2 = nn.Linear(40, 21)

    def forward(self, x):
        shape = x.shape
        x = x.view(-1, shape[-3], shape[-2], shape[-1])
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.hswish1(x)
        #x = self.se1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.hswish2(x)
        #x = self.se1(x) # ?
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.hswish2(x)
        #x = self.se1(x)
        
        x = self.conv4(x)
        x = self.bn2(x)
        x = self.hswish2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1) # Flatten the tensor
        
        if self.predict:
            x = self.fc1(x)
            x = self.drop(x)
            x = self.hswish3(x)
            x = self.fc2(x)
            x = nn.functional.softmax(x, dim=0)
            
        x = x.view(shape[0], -1, x.shape[-1])
        x = x.squeeze()
        
        return x