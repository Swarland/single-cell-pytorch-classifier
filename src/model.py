import torch.nn as nn

class SingleCellModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(SingleCellModel, self).__init__()

        #create first later
        self.fc1 = nn.Linear(num_features, 256)
        self.relu1 = nn.ReLU()

        #dropout to help avoid overfitting

        self.dropout = nn.Dropout(p=0.3) 

        #create second layer
        self.fc2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU()

        #output layer
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
