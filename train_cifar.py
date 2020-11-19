from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import xlrd
import os
import argparse
#from read_ImageNetData import ImageNetData
from CIFAR100 import ImageNetData
#from corloss1 import Correlation_CrossEntropyLoss#w
from corloss2 import Correlation_CrossEntropyLoss#s
import se_resnet
import se_resnext
from tensorboardX import SummaryWriter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataset_sizes):
    since = time.time()
    resumed = False
    #Xid  Yid
    a1 = torch.empty(1,100)
    a2 = torch.empty(1,100)
    b1 = torch.empty(1,100)
    b2 = torch.empty(1,100)
    file_name = 'resnext_cifar100.xlsx'#senet
    #file_name = 'prob_cifar.xlsx'#resnet
    file = xlrd.open_workbook(file_name)
    sheet = file.sheet_by_name('Sheet3')
    nrows = sheet.nrows
    for i in range(nrows):
        cell_value_x1 = sheet.cell_value(i,1)
        cell_value_y1 = sheet.cell_value(i,2)
        cell_value_x2 = sheet.cell_value(i,3)
        cell_value_y2 = sheet.cell_value(i,4)
        a1[0,i] = cell_value_x1
        b1[0,i] = cell_value_y1
        a2[0,i] = cell_value_x2
        b2[0,i] = cell_value_y2
    X1 = a1
    Y1 = b1
    X2 = a2
    Y2 = b2
        
   

    best_model_wts = model.state_dict()

    for epoch in range(args.start_epoch+1,num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                if args.start_epoch > 0 and (not resumed):
                    scheduler.step(args.start_epoch+1)
                    resumed = True
                    
                else:
                    scheduler.step(epoch)
                
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            correct_count_5 = 0.0
            yuan_running_loss = 0.0
            K = 0.0
            Z = 0.0
            J = 0.0
            prob_corloss = 0.0
            aver_z = 0.0


            tic_batch = time.time()
            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloders[phase]):
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                #print(outputs.size())
                maxk = max((1,5))#top-5
                _, preds = torch.max(outputs.data, 1)
                _, pred_5 = outputs.topk(maxk, 1, True, True)#top-5
                loss = criterion(outputs, labels)
                yuanloss = criterion(outputs, labels)#celoss
                #print(loss)
                cor_output = outputs.cpu()
                target = labels
                #a = cor_criterion(cor_output, target, X1,Y1,T=3000)#d=1
                a = cor_criterion(cor_output, target, X1,Y1,X2,Y2,T=3000)#d=2
                loss1 = a[0]
                loss2 = loss1.view(1)
                corloss = loss2.squeeze(0)
                corloss = corloss.cuda()
                #print(corloss)
                loss.data = corloss
                #print(loss)
                #outputs = model(inputs)
                #_, preds = torch.max(outputs.data, 1)
                #loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                yuan_running_loss += yuanloss.item()
                running_loss += loss.item()
                K += a[1]
                Z += a[2]
                J += a[3]
                #running_loss += loss.data[0]
                running_corrects += float(torch.sum(preds == labels.data))
                correct_count_5 += float(torch.sum(pred_5[:,0] ==labels.data))
                correct_count_5 += float(torch.sum(pred_5[:,1] ==labels.data))
                correct_count_5 += float(torch.sum(pred_5[:,2] ==labels.data))
                correct_count_5 += float(torch.sum(pred_5[:,3] ==labels.data))
                correct_count_5 += float(torch.sum(pred_5[:,4] ==labels.data))#top-5
                #running_corrects += torch.sum(preds == labels.data)

                batch_loss = running_loss / ((i+1)*args.batch_size)
                batch_acc = running_corrects / ((i+1)*args.batch_size)
                batch_acc_5 = correct_count_5 / ((i+1)*args.batch_size)
                #batch_K = K / ((i+1)*args.batch_size)
                #print(batch_K)

                if phase == 'train' and i%args.print_freq == 0:
                    print('[Epoch {}/{}]-[batch:{}/{}] lr:{:.4f} {} Loss: {:.6f}  Acc: {:.4f}  top-5:{:.4f}  Time: {:.4f}batch/sec'.format(
                          epoch, num_epochs - 1, i, round(dataset_sizes[phase]/args.batch_size)-1, scheduler.get_lr()[0], phase, batch_loss, batch_acc,batch_acc_5,args.print_freq/(time.time()-tic_batch)))
                    tic_batch = time.time()

            if phase == 'train':
                epoch_loss = running_loss / dataset_sizes['train']            
                prob_corloss = K / dataset_sizes['train']
                prob_loss = J / dataset_sizes['train']
                if K != 0:
                    aver_z = Z / K
                else:
                    aver_z = 0
                
            else:
                epoch_loss = yuan_running_loss / dataset_sizes['val']
            #epoch_loss = running_loss / dataset_sizes[phase]#CELOSS
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_acc_5 = correct_count_5 / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f} top-5: {:.4f}  '.format(
                phase, epoch_loss, epoch_acc, epoch_acc_5))

            if phase == 'train':
                writer.add_scalar('train loss',epoch_loss,global_step = epoch)
                writer.add_scalar('train acc',epoch_acc,global_step = epoch)
                writer.add_scalar('prob_corloss',prob_corloss,global_step = epoch)
                writer.add_scalar('prob_loss',prob_loss,global_step = epoch)
                writer.add_scalar('aver_z',aver_z,global_step = epoch)
            elif phase == 'val':
                writer.add_scalar('val loss',epoch_loss,global_step = epoch)
                writer.add_scalar('val acc',epoch_acc,global_step = epoch)
                writer.add_scalar('top-5',epoch_acc_5,global_step = epoch)
        writer.close()

        if (epoch+1) % args.save_epoch_freq == 0:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model, os.path.join(args.save_path, "epoch_" + str(epoch) + ".pth.tar"))

           


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
    parser.add_argument('--data-dir', type=str, default="./ImageData")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-class', type=int, default=100)
    parser.add_argument('--num-epochs', type=int, default=71)
    parser.add_argument('--lr', type=float, default=0.01)#0.045
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default="output/resnext101/d=2/3000.1cifar100")
    parser.add_argument('--resume', type=str, default=None, help="For training from one checkpoint")
    parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
    parser.add_argument('--network', type=str, default="se_resnet_101", help="")
    args = parser.parse_args()

    # read data
    dataloders, dataset_sizes = ImageNetData(args)
    #print(dataset_sizes['val'])

    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # get model
    print('args.network=',args.network)
    #script_name = '_'.join([args.network.strip().split('_')[0], args.network.strip().split('_')[1]])
    script_name = args.network.strip().split('_')[0]
    print(script_name)

    if script_name == "se_resnet":
        model = getattr(se_resnet ,args.network)(num_classes = args.num_class)
    elif script_name == "se_resnext":
        model = getattr(se_resnext, args.network)(num_classes=args.num_class)
    elif script_name == "resnext":
        from torchvision import models
        model = models.resnext101_32x8d(pretrained = False)
    else:
        raise Exception("Please give correct network name such as se_resnet_xx or se_rexnext_xx")

    
    print('args.resume=',args.resume)
    
    if args.resume:
        
        #if isfile(args.resume):
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            
            base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.state_dict().items())}
            
            model.load_state_dict(base_dict)
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
    if use_gpu:
        model = model.cuda()
        #model = torch.nn.DataParallel(model, device_ids=[0])
        model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])

    # define loss function
    criterion = nn.CrossEntropyLoss()
    cor_criterion = Correlation_CrossEntropyLoss()

    #Visualization path
    writer = SummaryWriter('/home/lj/External/SENet-PyTorch-master/runs/resnext101/d=2/3000.1cifar100')

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00004)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.98)

    model = train_model(args=args,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=args.num_epochs,
                           dataset_sizes=dataset_sizes)
