from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston=load_boston()
X,y=(boston.data,boston.target)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
myset=list(zip(X_train,y_train))  #注意，定义为Dataloader接收的形式

from torch.utils import data
import torch

dtype=torch.FloatTensor
train_loader=data.DataLoader(myset,batch_size=128,shuffle=True)

import torch.nn as nn
import torch.nn.functional as F
class Net1(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Net1, self).__init__()
        self.layer1=torch.nn.Sequential(nn.Linear(in_dim,n_hidden_1))
        self.layer2=torch.nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2))
        self.layer3=torch.nn.Sequential(nn.Linear(n_hidden_2,out_dim))

    def forward(self,x):
        x1=F.relu(self.layer1(x))
        x1=F.relu(self.layer2(x1))
        x2=self.layer3(x1)


#parallel model
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=Net1(13,16,32,1)
if torch.cuda.device_count()>1:
    print("let's use",torch.cuda.device_count(),'GPUs')
    model=nn.DataParallel(model)
model.to(device)

#optim&loss
optimizer_orig=torch.optim.Adam(model.parameters(),lr=0.01)
loss_func=torch.nn.MSELoss()

for epoch in range(100):
    model.train()
    for data,label in train_loader:
        input=data.type(dtype).to(device)
        label=label.type(dtype).to(device)
        output=model(input)
        loss=loss_func(output,label)
        optimizer_orig.zero_grad()
        loss.backward()
        optimizer_orig.step()
