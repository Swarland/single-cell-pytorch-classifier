import torch.nn as nn

class SingleCellModel(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size1 = 256, hidden_size2 = 64, dropout_rate = 0.3):
        super(SingleCellModel, self).__init__()

        #create first later
        self.fc1 = nn.Linear(num_features, hidden_size1)
        self.relu1 = nn.ReLU()

        #dropout to help avoid overfitting

        self.dropout = nn.Dropout(dropout_rate) 

        #create second layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()

        #output layer
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
