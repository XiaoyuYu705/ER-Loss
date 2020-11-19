from __future__ import print_function
from torch.autograd import Variable
import torch 
import xlrd
import torch.nn as nn
import numpy as np

class Correlation_CrossEntropyLoss(nn.Module):
    def __init__(self,weight=None,size_average=True):
        super(Correlation_CrossEntropyLoss,self).__init__()
        self.weight = weight
        self.size_average = size_average

    def forward(self,input,target,X1,Y1,X2,Y2,T):
        batch_loss = 0.
        batch_k = 0.0
        batch_z = 0.0
        batch_j = 0.0
        
        inputs = input.data
     
        for i in range(inputs.shape[0]):
            #print(inputs[i])
            #print(target[i])
            #print(inputs[i,target[i]])
            numerator = np.exp(inputs[i,target[i]])
            A = np.exp(inputs[i, :])
            #print(A)
            denominator = A.sum(dim=0)
            target_prob = numerator / denominator
            index = torch.tensor([[target[i]]])
            #print(index)

            Xid1 = X1.gather(1,index)
            Xid2 = X2.gather(1,index)#d=2

            Yid1 = Y1.gather(1,index)
            Yid1 = Yid1.view(1)
            Yid1 = Yid1.squeeze(0).long()
            #print(Yid1)
            #Yid1.data = Yid1.data-1#ImageNet
            #print(Yid1)
            Yid2 = Y2.gather(1,index)#d=2
            Yid2 = Yid2.view(1)#d=2
            Yid2 = Yid2.squeeze(0).long()#d=2
            #Yid2.data = Yid2.data-1#ImageNet
            
            cor_1 = inputs[i].gather(0,Yid1)#outputsd的值
            #print(cor_1)
            Pid1 = np.exp(cor_1) / denominator
            cor_2 = inputs[i].gather(0,Yid2)#output的值,d=2
            Pid2 = np.exp(cor_2) / denominator#d=2
            #print(Pid2)
        
            k = 0
            z = 0
            j = 0
            if target_prob > (T * ((Xid1 * Pid1)+(Xid2 * Pid2))):#d=2
            #if target_prob > (T * (Xid1 * Pid1)):#d=1
                loss = -np.log(target_prob - T * ((Xid1 * Pid1)+(Xid2 * Pid2)))#d=2
                #print(target_prob - T * ((Xid1 * Pid1)+(Xid2 * Pid2)))
                #print(loss)
                #loss = -np.log(target_prob - T * (Xid1 * Pid1))#d=1
                if Pid1 != 0 or Pid2 != 0:#d=2
                #if Pid1 != 0:#d=1
                    k = k+1
                    z = ((target_prob) / (T * ((Xid1 * Pid1)+(Xid2 * Pid2))))#d=2
                    #z = (target_prob / (T * (Xid1 * Pid1)))#d=1
                else:
                    k = k+0
            else:
                loss = -np.log(target_prob)
                #print(loss)
                j = j+1
            #print(loss)
               
            batch_loss = batch_loss + loss
            batch_k = batch_k + k
            batch_z = batch_z + z
            batch_j = batch_j + j
        #batch loss average
        if self.size_average == True:
            batch_loss /= inputs.shape[0]
              
        return batch_loss, batch_k, batch_z, batch_j
    
    
    
