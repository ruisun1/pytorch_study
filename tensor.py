import torch

#y=torch.index_select(input,dim,index)

input=torch.LongTensor([[1,2,3],[4,5,6]])
index=torch.LongTensor([0,1])
y=torch.index_select(input=input,dim=1,index=index)  #选取第0列，第1列
print(y)


#tensor([[1, 2],
#        [4, 5]])

input=torch.LongTensor([[1,2,3],[4,5,6]])
index=torch.LongTensor([[0,2],[0,1]])   #以行为顺序，对每行，按列索引
y=torch.gather(input=input,dim=1,index=index)
print(y)
#tensor([[1, 3],
 #       [4, 5]])

input=torch.LongTensor([[0,1,1],[1,0,0]])
y=torch.nonzero(input)     #输出非零索引
print(y)

#tensor([[0, 1],
#        [0, 2],
#        [1, 0]])


input=torch.LongTensor([[1,2,3],[4,5,6]])
mask=torch.BoolTensor([[False,True,True],[True,False,False]])
y=torch.masked_select(input,mask)    #输出bool值为True的值
print(y)
#tensor([2, 3, 4])

input=torch.LongTensor([[1,2,3,4],[5,6,7,8]])    #将tensor限制到min和max之间
y=torch.clamp(input=input,min=1,max=3)
print(y)
#tensor([[1, 2, 3, 3],[3, 3, 3, 3]])

input=torch.LongTensor([[1,2,3,4],[5,6,7,8]])
y=torch.sigmoid(input)
print(y)
#tensor([[0.7311, 0.8808, 0.9526, 0.9820],[0.9933, 0.9975, 0.9991, 0.9997]])


input=torch.LongTensor([[1,2,3,4],[5,6,7,8]])
#y=torch.exp(input)
y=torch.pow(input,2)
print(y)
#tensor([[ 1,  4,  9, 16],[25, 36, 49, 64]])

input=torch.LongTensor([[1,2,3,4],[5,6,7,8]])
y=torch.cumprod(input,dim=0) #累乘
y=torch.cumsum(input,dim=0)#类加

input=torch.tensor([[1,2,3,4],[5,6,7,8]])
input=input.float()
y=torch.mean(input,dim=0,keepdim=True)
print(y)
#tensor([[3., 4., 5., 6.]])

input=torch.tensor([[1,2,3,4],[5,6,7,8]])
input2=torch.tensor([[2,2,3,4],[3,3,3,5]])
print(torch.eq(input,input2))
#tensor([[False,  True,  True,  True],[False, False, False, False]])

input=torch.tensor([[1,2,3,4],[5,6,7,8]])
y=torch.max(input,dim=0)  #结果的index值为行标
print(y)
#torch.return_types.max(
#values=tensor([5, 6, 7, 8]),
#indices=tensor([1, 1, 1, 1]))
values,indices=torch.max(input,dim=0)


input=torch.tensor([[1,2,3,4],[5,6,7,8]])
y=torch.topk(input,dim=1,k=2)
print(y)
#torch.return_types.topk(
#values=tensor([[4, 3],[8, 7]]),
#indices=tensor([[3, 2],[3, 2]]))
