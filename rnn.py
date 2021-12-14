#PyTorch提供了两个版本的循环神经网络接口，单元版的输入是每个时间步，或循环神经网络的一个循环，而封装版的是一个序列。

#封装版torch.nn.RNN
#nn.RNN 参数说明
#num_layers：RNN层数，nonlinearity：指定非线性函数使用tanh还是relu。默认是tanh。bias：如果是False，那么RNN层就不会使用偏置权重bi和bh，默认是 True。·batch_first：如果True的话，那么输入Tensor的shape应该是 （batch,seq,feature），输出也是这样。默认网络输入是 （seq,batch,feature），即序列长度、批次大小、特征维度。
#·dropout：如果值非零（该参数取值范围为0～1之间），那么除了最后 一层外，其他层的输出都会加上一个dropout层，缺省为零。bidirectional：如果True，将会变成一个双向RNN，默认为False。
#函数nn.RNN()的输入包括特征及隐含状态，记为（xt、h0)，输出包括输 出特征及输出隐含状态，记为（outputt、hn)。

import torch
import torch.nn as nn
rnn=nn.RNN(input_size=10,hidden_size=20,num_layers=2)

#at=tanh(wih*xt+bih+whh*at-1+bhh)
#其中特征值xt的形状为（seq_len,batch,input_size），h0的形状为 （num_layers*num_directions,batch,hidden_size），其中num_layers为层数， num_directions方向数，如果取2则表示双向（bidirectional,），取1则表示单 向。outputt的形状为（seq_len,batch,num_directions*hidden_size），hn的形状 为（num_layers*num_directions,batch,hidden_size）。
input=torch.randn(100,32,10) #(seq_len,batch_size,feature_size)特征长度为100，批量大小为32，特征维度为10
h_0=torch.randn(2,32,20)
output,h_n=rnn(input,h_0)
print(output.shape,h_n.shape)




#RNNCell， RNNCell的输入形状为(batch,input_size)，隐含状态输入：(batch,hidden_size),网络输出只有隐含状态输出,:(batch,hdden_size)

#pytorch 实现RNN
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RNN,self).__init__()
        self.hidden_size=hidden_size
        self.i2h=nn.Linear(input_size+hidden_size,hidden_size)
        self.i2o=nn.Linear(input_size+hidden_size,output_size)
        self.softmax=nn.LogSoftmax(dim=1)
    def forward(self,input,hidden):
        combined=torch.cat((input,hidden),1)
        hidden=self.i2h(combined)
        output=self.i2o(combined)
        output=self.softmax(output)
        return output,hidden

    def initHidden(self):
        return torch.zeros(1,self.hidden_size)





