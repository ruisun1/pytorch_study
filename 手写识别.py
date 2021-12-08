import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
from torchvision.datasets import mnist
#预处理模块
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#导入nn及优化器
import torch.nn.functional as F
import torch.optim as optim
from torch import  nn

#定义超参数
train_batch_size=64
test_batch_size=128
learning_rate=0.01
num_epoches=20
momentum=0.5

#下载数据并预处理
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])  #Normalize表示对张量归一化[0,5],[0.5]分别表示均值和方差

train_dataset=mnist.MNIST('./data',train=True,download=False,transform=transform)
test_dataset=mnist.MNIST('./data',train=False,download=False,transform=transform)

train_loader=DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=test_batch_size,shuffle=False)

examples=enumerate(test_loader)
batch_idx,(example_data,example_targets)=next(examples)

fig=plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0],cmap='gray',interpolation='none')
    plt.title("Ground Truth:{}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])

plt.show()

class Net(nn.Module):
    def __init__(self,in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        self.layer1=nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.BatchNorm1d(n_hidden_1))
        self.layer2=nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.BatchNorm1d(n_hidden_2)) #batchnorm维度等于前一层输出维度
        self.layer3=nn.Sequential(nn.Linear(n_hidden_2,out_dim))

    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=self.layer3(x)
        return x

#实例化网络
device=torch.device("cude:0" if torch.cuda.is_available() else "cpu")
model=Net(28*28,300,100,10)
model.to(device)

#定义损失函数
criteron=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=learning_rate ,momentum=momentum)

#开始训练
losses=[]
acces=[]

eval_losses=[]
eval_acces=[]
for epoch in range(num_epoches):
    train_loss=0
    train_acc=0
    model.train()
    #动态修改学习率
    if epoch%5==0:
        optimizer.param_groups[0]['lr']*=0.1
        #optimizer.param_groups： 是长度为2的list，其中的元素是2个字典
        #optimizer.param_groups[0]： 长度为6的字典，包括[‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]这6个参数；
    for img,label in train_loader:
        img=img.to(device)
        label=label.to(device)
        img=img.view(img.size(0),-1)
        out=model(img)
        loss=criteron(out,label)  #nn.crossentropy 包含softmax和nlloose两个过程，所以不用事先softmax
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() #更新所有参数
        train_loss+=loss.item()
        _,pred=out.max(dim=1)     #第一个返回值是值，第二个是index
        print(pred)
        num_correct=(pred==label).sum().item()
        acc=num_correct/img.shape[0]
        train_acc+=acc
    losses.append(train_loss/len(train_loader))
    acces.append(train_acc/len(train_loader))

    #检验eval
    eval_loss=0
    eval_acc=0
    model.eval()
    for img,label in test_loader:
        img=img.to(device)
        label=label.to(device)
        img=img.view(img.size(0),-1)
        out=model(img)
        loss=criteron(out,label)
        eval_loss+=loss.item()
        _,pred=out.max(dim=1)
        num_correct=(pred==label).sum().item()
        acc=num_correct/img.shape[0]
        eval_acc+=acc

    eval_losses.append(eval_loss/len(test_loader))
    eval_acces.append(eval_acc/len(test_loader))
    print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}' .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader), eval_loss / len(test_loader), eval_acc / len(test_loader)))




