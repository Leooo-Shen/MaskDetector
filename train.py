import os
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from nets.yolo_training import YOLOLoss,Generator
from nets.yolo4 import YoloBody

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataloader import YoloDataset, yolo_dataset_collate

print("Cuda is available:", torch.cuda.is_available())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1,3,2])[::-1,:,:]

def fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    total_loss = 0
    val_loss = 0
    for iteration in range(epoch_size):
        start_time = time.time()
        images, targets = next(gen)
        with torch.no_grad():
            if cuda:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            else:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
        # print(images)
        optimizer.zero_grad()
        outputs = net(images)
        losses = []
        for i in range(3):
            loss_item = yolo_losses[i](outputs[i], targets)
            losses.append(loss_item[0])
        loss = sum(losses)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        total_loss += loss
        waste_time = time.time() - start_time
        # print("Epoch:",epoch)
        if iteration == 0 or (iteration+1) % 10 == 0:
            print("Epoch: " + str(epoch+1) + '|| step:' + str(iteration+1) + '/' + str(epoch_size) + ' || Total Loss: %.4f || %.4fs/step' % (total_loss/(iteration+1), waste_time))

    if epoch == 0 or (epoch + 1) % 20 == 0:

        print('Start Validation')
        for iteration in range(epoch_size_val):
            images_val, targets_val = next(genval)

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                for i in range(3):
                    loss_item = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item[0])
                loss = sum(losses)
                val_loss += loss
        print('Finish Validation')
        print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

        print('Saving state, epoch:', str(epoch+1))
        torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))



if __name__ == "__main__":

    ####### configs ########

    input_shape = (416,416)
    # input_shape = (608, 608)
    Cosine_lr = True
    mosaic = True
    Cuda = True
    smoooth_label = 0.005
    Freeze_Backbone = True
    Use_Data_Loader = True

    annotation_path = '2007_train.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    classes_path = 'model_data/mask_classes.txt'
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)
    

    model = YoloBody(len(anchors[0]),num_classes)
    model_path = "model_data/yolo4_voc_weights.pth"

    print('Loading weights into state dict...', model_path)
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_path)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # print('Finished!')

# 修改网络结构后对齐预权值
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    a = {}
    for k, v in pretrained_dict.items():
        try:
            if np.shape(model_dict[k]) == np.shape(v):
                a[k] = v
        except:
            pass
    # print(a)
    model_dict.update(a)
    model.load_state_dict(model_dict)
    print('Finished!')


    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()


    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(anchors,[-1,2]),num_classes, \
                                (input_shape[1], input_shape[0]), smoooth_label, Cuda))

    # 0.1 for val, 0.9 for training
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    if Freeze_Backbone:
        lr = 1e-3
        Batch_size = 4
        Init_Epoch = 0
        Freeze_Epoch = 50
        
        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)


        gen = Generator(Batch_size, lines[:num_train],
                        (input_shape[0], input_shape[1])).generate(mosaic = mosaic)
        gen_val = Generator(Batch_size, lines[num_train:],
                        (input_shape[0], input_shape[1])).generate(mosaic = False)

        epoch_size = int(max(1, num_train//Batch_size//2.5)) if mosaic else max(1, num_train//Batch_size)
        epoch_size_val = num_val//Batch_size

        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda)
            lr_scheduler.step()

    if True:
        lr = 1e-4
        Batch_size = 2
        Freeze_Epoch = 50
        Unfreeze_Epoch = 300

        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)

        gen = Generator(Batch_size, lines[:num_train],
                        (input_shape[0], input_shape[1])).generate(mosaic = mosaic)
        gen_val = Generator(Batch_size, lines[num_train:],
                        (input_shape[0], input_shape[1])).generate(mosaic = False)
                        
        epoch_size = int(max(1, num_train//Batch_size//2.5)) if mosaic else max(1, num_train//Batch_size)
        epoch_size_val = num_val//Batch_size

        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda)
            lr_scheduler.step()
