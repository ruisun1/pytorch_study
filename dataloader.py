import torch
import torchvision.transforms
from torch.utils import data
import  numpy as np

#定义数据类，继承dataset
class TestDataset(data.Dataset):
    def __init__(self):
        self.Data=np.asarray([[1,2],[3,4],[2,1],[3,4],[4,5]])
        self.Label=np.asarray([[0],[1],[0],[1],[2]])
    def __getitem__(self, index):
        txt=torch.Tensor(self.Data[index])
        label=torch.Tensor(self.Label[index])
        return txt,label
    def __len__(self):
        return len(self.Data)

Test=TestDataset()

print(Test[2]) # __getitem__(2)
print(Test.__len__())

#getitem 只能处理单个数据，如果想批量处理数据需要用DataLoader

Test2=TestDataset()
test_loader=data.DataLoader(Test2,batch_size=2,shuffle=False,num_workers=0)
for i,traindata in enumerate(test_loader):
    print("i:",i)
    Data,Label=traindata
    print("data:",Data)
    print("Label",Label)

