import torch
import torch.nn as nn
import numpy
import math
#import xlwt
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

import os

from PIL import Image, ImageDraw, ImageFont

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']



class ImageNetValDataSet(torch.utils.data.Dataset):
    def __init__(self, img_path, img_label,image_transforms):
        self.image_transforms = image_transforms
        img_names = os.listdir(img_path)
        img_names.sort()
        #img_names.sort(key = lambda x : int(x[:-4]))#in order
        self.img_path = [os.path.join(img_path, img_name) for img_name in img_names]
        with open(img_label,"r") as input_file:
            lines = input_file.readlines()
            #self.img_label = [(int(line)-1) for line in lines]
            self.img_label = [(int(line.strip())-1) for line in lines]

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        img = Image.open(self.img_path[item]).convert('RGB')
        label = self.img_label[item]
        if self.image_transforms is not None:
            try:
                img = self.image_transforms(img)
            except:
                print("Cannot transform image: {}".format(self.img_path[item]))
        return img, label
#data enhancement


#write in excel
#workbook = xlwt.Workbook(encoding = 'utf-8')
#worksheet = workbook.add_sheet('1')

def savetxt(filename,labelsum):
    with open(filename,'a') as file_handle:
        result = str(labelsum)
        file_handle.write(result)
        file_handle.write('\n')

#def test dataset acc 
def computeTestSetAccuracy(model):
    
    #GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_acc = 0.0
    test_loss = 0.0
    correct_counts = 0.0#yxy0411
    correct_count_5=0.0#yxy0412
    with torch.no_grad():
        model.eval()
        for j, (inputs, labels) in enumerate(test_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            SUM = 0
            for i in range(100):
                label_i = outputs[:,i]
                label_sum = label_i.sum(dim=0)
                label_aver = label_sum.item()/350
                label_sum_exp = math.exp(label_aver)
                #print(label_i)
                #print(label_sum.item())
                #savetxt('sum1.txt',label_sum.item())
                #worksheet.write(i,0,label = i)
                #worksheet.write(i,1,label = label_sum.item())
                SUM = SUM+label_sum_exp

            sums = SUM
            #print(sums)
            for i in range(100):
                label_i = outputs[:,i]
                label_sum = label_i.sum(dim=0)
                label_aver = label_sum.item()/350
                label_sum_exp = math.exp(label_aver)
                prob = label_sum_exp/sums
                savetxt('worm.txt',prob)
                #print(format(prob,'.8f'))
                #worksheet.write(i,2,label = prob)
            #workbook.save('gailv.xls')

image_transforms = {
    'test': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

#load test data
dataset = 'ImageData'

data = {}
data['test'] = ImageNetValDataSet(os.path.join(dataset, 'cifar100','worm'),
                                               os.path.join(dataset, 'ILSVRC2012_devkit_t12', 'data','cifar100','worm.txt'),image_transforms['test'])
num_classes = 100#what is this???
batch_size = 350
test_data = DataLoader(data['test'],
                       batch_size=350,
                       shuffle=True)
#test_data_size = len(data['test'])

if __name__ == '__main__':
    model = torch.load('output/epoch_70.pt')
    
    computeTestSetAccuracy(model)
