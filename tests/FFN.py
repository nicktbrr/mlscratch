import os
import sys
import torch
import pandas as pd
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from DL.layers.linear import linear
from activations.activations import ReLu

class FFN(torch.nn.Module):
    def __init__(self, input_shape):
        super(FFN, self).__init__()
        self.in_shape = input_shape
        self.l1 = linear(input_shape, 100)
        self.l2 = linear(100, 100)
        self.l3 = linear(100, 1)
    
    def forward(self, x):
        x = self.l1(x)
        x = ReLu(x)
        x = self.l2(x)
        x = ReLu(x)
        x = self.l3(x)
        return x




df = pd.read_csv('data/housing.csv', header=None, delimiter=r'\s+')
X = df.iloc[:,:-1]
y_true = df.iloc[:,-1]

X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_true_tensor = torch.tensor(y_true.values, dtype=torch.float32).view(-1, 1)

input_features = X_tensor.shape[1]
model = FFN(input_features)

y_pred = model(X_tensor)
print(y_pred.shape)