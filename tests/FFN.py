import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from DL.layers.linear import linear
from activations.activations import ReLu
from losses.losses import MSE
from optimizers.optimizers import SGD

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_true_tensor = torch.tensor(y_true.values, dtype=torch.float32).view(-1, 1)

input_features = X_tensor.shape[1]
model = FFN(input_features)

y_pred = model(X_tensor)
loss = MSE(y_pred, y_true_tensor)

model.train()
optim = SGD(model.parameters(), lr=0.0001)

epochs = 3000
progress_bar = tqdm(range(epochs), desc="Training Progress")

for _ in progress_bar:
    y_pred = model(X_tensor)
    loss = MSE(y_pred, y_true_tensor)
    loss.backward()
    optim.step()
    optim.zero_grad()
    progress_bar.set_postfix({'loss': f'{loss:.4f}'})


