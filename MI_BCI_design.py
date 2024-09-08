import matplotlib.pyplot as plt
import numpy as np
import torch
import sklearn.metrics as metrics
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import copy
import time
from scipy.io import savemat
from scipy import signal

def load_sbjnum(num,data,labels):
    """
    get data and labels of the 'num'th subject
    """
    return data[num], labels[num]

def trail_plot(trail,label,channelname,xaxis=np.array([])):
    """
    draw the 22 channel data plot for one trail
    """
    # plot raw data for each trail
    label_name = {'0':'left','1':'right','2':'foot','3':'tongue'}
    plt.title(f'{label_name[str(label)]}')
    ax = plt.gca() # remove y-axis scale value
    ax.axes.yaxis.set_ticks([]) # remove y-axis scale value
    if xaxis.size == 0:
        xaxis = np.linspace(1, trail.shape[-1], trail.shape[-1])
    for i in range(trail.shape[0]):
        plt.subplot(trail.shape[0],1,i+1)
        plt.box() # remove border of subplot
        plt.plot(xaxis,trail[i])
        plt.ylabel(channelname[i],labelpad=15,rotation=0,fontsize=6) # labelpad is label and axis distance
        plt.xticks([])
        plt.yticks([])
        # plt.subplots_adjust(left=0,right=1)
    plt.subplots_adjust(hspace=0) # remove subplots distance, hspace is subplots hight distance
    plt.show()

class EEGdata(Dataset):
    """
    put data and labels in to the class to facilitate deep learning
    """
    def __init__(self,data,label):
        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(label)
        self.len = label.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len

class EEGNet(nn.Module):
    """
    EEGNet
    Args:
        C (_type_): number of eeg channels
        T (_type_): length of eeg signal
        F (_type_): base number of filters (number of convolution feature channels)
        D (_type_): number of depthwise convolution mulitpler
        N (_type_): number of classes
    """
    def __init__(self, C, T, F, D, N):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F, kernel_size=(1,125),bias=False,padding='same'),
            nn.BatchNorm2d(F)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F, F*D, kernel_size=(C,1), groups=F,bias=False),# default padding = 'valid'
            nn.BatchNorm2d(F*D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,8)),
            nn.Dropout(p=0.5)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F*D, F*D, kernel_size=(1,16), groups=F*D, bias=False, padding='same'),
            nn.Conv2d(F*D, F*D, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(F*D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,8)),
            nn.Dropout(p=0.5),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(F*D*(T//64),N,bias=False)
        )
    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x
    
class EEGNeX(nn.Module):
    """
    EEGNeX
    Args:
        C (_type_): number of eeg channels
        T (_type_): length of eeg signal
        F (_type_): base number of filters (number of convolution feature channels)
        D (_type_): number of depthwise convolution mulitpler
        N (_type_): number of classes
    """
    def __init__(self, C, T, F, D, N):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F, kernel_size=(1,32),bias=False,padding='same'),
            nn.BatchNorm2d(F)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F, F*4, kernel_size=(1,32),bias=False,padding='same'),
            nn.BatchNorm2d(F*4),
            nn.ELU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F*4, F*4*D, kernel_size=(C,1), groups=F*4,bias=False),
            nn.BatchNorm2d(F*4*D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,4)),
            nn.Dropout(p=0.5)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(F*4*D, F*4*D, kernel_size=(1,16),bias=False,padding='same',dilation=(1,2)),
            nn.BatchNorm2d(F*4*D)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(F*4*D, F, kernel_size=(1,16),bias=False,padding='same',dilation=(1,4)),
            nn.BatchNorm2d(F),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,8)),
            nn.Dropout(p=0.5),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(248,N)
        )
    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # print(x.shape)
        x = self.classifier(x)
        return x

class EEGNext(nn.Module):
    """
    EEGNext
    Args:
        C (_type_): number of eeg channels
        T (_type_): length of eeg signal
        F (_type_): base number of filters (number of convolution feature channels)
        D (_type_): number of depthwise convolution mulitpler
        N (_type_): number of classes
    """    
    def __init__(self, C, T, F, D, N):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F, kernel_size=(1,32),bias=False,padding='same'),
            nn.BatchNorm2d(F)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F, F*4, kernel_size=(1,32),bias=False,padding='same'),
            nn.BatchNorm2d(F*4),
            nn.ELU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F*4, F*4*D, kernel_size=(C,1), groups=F*4,bias=False),
            nn.BatchNorm2d(F*4*D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,4)),
            nn.Dropout(p=0.5)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(F*4*D, F*4*D, kernel_size=(1,16), groups=F*4*D, bias=False, padding='same'),
            nn.Conv2d(F*4*D, F*D, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(F*D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,8)),
            nn.Dropout(p=0.5),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(496,N)
        )
    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x

class weightConstraint(object):
    """
    to limit the paramaters of one part of deep network
    """
    def __init__(self,max):
        self.max = max
    def __call__(self,module):
        if hasattr(module,'weight'):
            print("Entered")
            w=module.weight.data
            w=torch.clamp_max(w,self.max) # clamp_max: only give upper limit
            module.weight.data=w

def show_plot(accuracy_history,loss_history,test_accuracy):
    plt.figure(figsize=(10,5))
    #fig2
    plt.subplot(121)
    plt.plot(loss_history,marker=".",color="c")
    plt.xlabel("epochs")
    plt.ylabel("loss value")
    plt.title('train loss')
    #fig3
    plt.subplot(122)
    plt.plot(accuracy_history,marker="o",label="train_acc") #plt.plot(x,y)定义x，y轴数据，定义颜色，标记型号，大小等
    plt.plot(test_accuracy, marker='o', label="test_acc")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("ACC")
    plt.legend(loc="best")
    plt.savefig('acc_loss.png')
    plt.show()
    
def plot_recall(epoch_list,recall1,recall2,recall3=None,recall4=None,N=4):
    plt.figure(figsize=(8,5))
    if N == 4:
        plt.plot(epoch_list,recall1, color='purple', label='Back1_Recall',marker=".")
        plt.plot(epoch_list,recall2,color='c',label="Back2_Recall",marker=".")
        plt.plot(epoch_list,recall3,color='g',label="Back3_Recall",marker=".")
        plt.plot(epoch_list,recall4,color='m',label="Back4_Recall",marker=".")
    elif N == 2:
        plt.plot(epoch_list,recall1, color='purple', label='Lefthand_Recall',marker=".")
        plt.plot(epoch_list,recall2,color='c',label="Righthand_Recall",marker=".")
    plt.title('Recall during test')
    plt.xlabel('Epoch')
    plt.ylabel('Recall_Vales')
    plt.legend()
    plt.savefig("recall.jpg")
    plt.show()

def plot_precision(epoch_list,precision1,precision2,precision3=None,precision4=None,N=4):
    plt.figure(figsize=(8,5))
    if N == 4:
        plt.plot(epoch_list,precision1, color='black', label='Back1_Precision',marker="o")
        plt.plot(epoch_list,precision2, color='b', label='Back2_Precision',marker="o")
        plt.plot(epoch_list,precision3, color='m', label='Back3_Precision',marker="o")
        plt.plot(epoch_list,precision4, color='c', label='Back4_Precision',marker="o")
    elif N == 2:
        plt.plot(epoch_list,precision1, color='black', label='Lefthand_Precision',marker="o")
        plt.plot(epoch_list,precision2, color='b', label='Righthand_Precision',marker="o")
    plt.xlabel('Epoch')
    plt.ylabel('Precision_Vales')
    plt.title('Precision during test')
    plt.legend()
    plt.savefig("precision.jpg")
    
    plt.show()

def plot_f1(epoch_list,f1_1,f1_2,f1_3=None,f1_4=None,N=4):
    plt.figure(figsize=(8,5))
    if N == 4:
        plt.plot(epoch_list,f1_1, color='yellow', label='Back1_F1',marker="^")
        plt.plot(epoch_list,f1_2, color='g', label='Back2_F1',marker="^")
        plt.plot(epoch_list,f1_3, color='b', label='Back3_F1',marker="^")
        plt.plot(epoch_list,f1_4, color='m', label='Back4_F1',marker="^")
    elif N == 2:
        plt.plot(epoch_list,f1_1, color='yellow', label='Lefthand_F1',marker="^")
        plt.plot(epoch_list,f1_2, color='g', label='Righthand_F1',marker="^")
    plt.xlabel('Epoch')
    plt.ylabel('F1_Values')
    plt.title('f1 during test')
    plt.legend()
    plt.savefig("f1.jpg")
    plt.show()

def DrawConfusionMatrix(save_model_name,C,T,F,D,N,val_dataloader):
    if save_model_name == 'EEGNetpara.pth':
        model = EEGNet(C,T,F,D,N)
    elif save_model_name == 'EEGNeXpara.pth':
        model = EEGNeX(C,T,F,D,N)
    else:
        model == EEGNext(C,T,F,D,N)
    model.load_state_dict(torch.load(save_model_name))
    model.eval()
    predict = []
    gt = []
    with torch.no_grad():
        for i, data in enumerate(val_dataloader,0):
                input = data[0].to(torch.float32)
                labels = data[1].long()
                input = input.unsqueeze(dim=1)
                output = eegnetmodel(input)
                _, predicted = torch.max(output.data, 1)
                pred = predicted
                y_true = labels.cpu()
                y_pred = pred.float().cpu()
                if len(predict)==0:
                    predict = y_pred.clone()
                    gt = y_true.clone()
                else:
                    predict = torch.concat((predict,y_pred))
                    gt = torch.concat((gt,y_true))
    cm = metrics.confusion_matrix(gt, predict)
    print(metrics.accuracy_score(gt,predict))
    print(metrics.classification_report(gt,predict))
    print(metrics.confusion_matrix(gt,predict))
    if N == 4:
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels = ['left','right','feet',"tongue"])
    elif N == 2:
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels = ['left','right'])
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()

def train(n_epochs, optimizer, model, loss_fn, train_loader):
    """
    define training function
    Args:
        n_epochs (_type_): total training cycle
        optimizer (_type_): 
        model (_type_): 
        loss_fn (_type_): 
        train_loader (_type_): 
    """
    
    loss_list = []
    accuracy_list = []
    for epoch in range(1,n_epochs+1):
        model.train()
        total = 0
        correct = 0
        Loss = 0
        for data in train_loader:
            input = data[0].to(torch.float32)
            labels = data[1].long()
            input = input.unsqueeze(dim=1) #(batch,eegchannel,time)-->(batcch,channels,eegchannel,time)
            output = model(input)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = loss_fn(output,labels)
            Loss += loss.item()
            # 注意在loss.backward()之前一定要将优化器的梯度置零，
            # 因为pytorch训练时优化器的梯度是累计的，在一批量迭代结束后，
            # 梯度并没有归零，如果不置零，则梯度会一直累加。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # torch.clamp_max(,1)
            # torch.clamp_max(,0.25)
        accuracy = correct/total
        Loss = Loss/(total/labels.size(0))
        loss_list.append(Loss)
        accuracy_list.append(accuracy)
        print("Traning: Epoch=%d, Loss=%.4f, Accuracy=%.4f"%(epoch, loss, accuracy))
        verify(eegnetmodel, val_loader)
    return loss_list, accuracy_list

def verify(model, val_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in val_loader:
            input = data[0].to(torch.float32)
            labels = data[1].long()
            input = input.unsqueeze(dim=1)
            output = model(input)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct/total
        print("Accuracy=%.4f"%(accuracy))

def normalization(train,test = np.array([])):
    """
    normalize all data of one subject
    Args:
        data (_type_): size(288,22,1000)
    """
    trainnum = train.shape[0]
    channelnum = train.shape[1]
    train = train.reshape(trainnum,-1)
    scaler = StandardScaler().fit(train) # x*=(x-μ)/σ, μ is mean, σ is Standard Deviation
    train = scaler.transform(train)
    train = train.reshape(trainnum,channelnum,-1)
    if test.size != 0:
        testnum = test.shape[0]
        test = test.reshape(testnum,-1)
        test = scaler.transform(test)
        test = test.reshape(testnum,channelnum,-1)
    return train,test

def classnum(test_data,dtype='Dataset'):
    '''
    calculate each class number of test data
    '''
    a = 0
    b = 0
    c = 0
    d = 0
    if dtype == 'Dataset':
        for i in range(len(test_data)):
            if test_data[i][1] == 0:
                a+=1
            elif test_data[i][1] == 1:
                b+=1
            elif test_data[i][1] == 2:
                c+=1
            elif test_data[i][1] == 3:
                d+=1
    else:
        for i in range(test_data.shape[0]):
            if test_data[i] == 0:
                a+=1
            elif test_data[i] == 1:
                b+=1
            elif test_data[i] == 2:
                c+=1
            elif test_data[i] == 3:
                d+=1
    print(a,b,c,d)

def buttferfiter(data):
    trailnum = data.shape[0]
    channelnum = data.shape[1]
    data = data.reshape(trailnum,-1)
    Fs = 250
    b, a = signal.butter(4, [7, 47], 'bandpass',fs=Fs)
    data = signal.filtfilt(b, a, data, axis=1)
    data = data.reshape(trailnum,channelnum,-1)
    return data

def cutdata(data,labels,time):
    # 定义2s的数据集
    partnum = int(4/time)
    timeidx = int(data.shape[-1]/partnum)
    trailidx = int(data.shape[0]*partnum)
    newdata = np.empty((trailidx,data.shape[1],timeidx))
    newlabels = np.empty((trailidx))
    for i in range(partnum):
        x = data[:,:,i*timeidx:(i+1)*timeidx]
        newdata[i*data.shape[0]:(i+1)*data.shape[0],:,:] = x
        newlabels[i*data.shape[0]:(i+1)*data.shape[0]] = labels
    return newdata,newlabels

def getcontrolsig(save_model_name,C,T,F,D,N,test_data):
    if save_model_name == 'EEGNetpara.pth':
        model = EEGNet(C,T,F,D,N)
    elif save_model_name == 'EEGNeXpara.pth':
        model = EEGNeX(C,T,F,D,N)
    else:
        model == EEGNext(C,T,F,D,N)
    val_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    model.load_state_dict(torch.load(save_model_name))
    model.eval()
    steplen = np.pi/180*5
    predict = []
    gt = []
    if N == 4:
        yaw = 0
        pitch = 0
        yaw_r = 0
        pitch_r = 0
        yaws = []
        pitchs = []
        yaws_r = []
        pitchs_r = []
        with torch.no_grad():
            for i, data in enumerate(val_dataloader,0):
                    input = data[0].to(torch.float32)
                    labels = data[1].long()
                    input = input.unsqueeze(dim=1)
                    output = eegnetmodel(input)
                    _, predicted = torch.max(output.data, 1)
                    if predicted == 0:
                        yaw += -steplen
                    elif predicted == 1:
                        yaw += steplen
                    elif predicted == 2:
                        pitch += -steplen
                    elif predicted == 3:
                        pitch += steplen
                    yaws.append([yaw]*1000)
                    pitchs.append([pitch]*1000)

                    if labels == 0:
                        yaw_r += -steplen
                    elif labels == 1:
                        yaw_r += steplen
                    elif labels == 2:
                        pitch_r += -steplen
                    elif labels == 3:
                        pitch_r += steplen
                    yaws_r.append([yaw_r]*1000)
                    pitchs_r.append([pitch_r]*1000)
                    pred = predicted
                    y_true = labels.cpu()
                    y_pred = pred.float().cpu()
                    if len(predict)==0:
                        predict = y_pred.clone()
                        gt = y_true.clone()
                    else:
                        predict = torch.concat((predict,y_pred))
                        gt = torch.concat((gt,y_true))
        yaws = np.array(yaws).reshape(-1)
        pitchs = np.array(pitchs).reshape(-1)
        yaws_r = np.array(yaws_r).reshape(-1)
        pitchs_r = np.array(pitchs_r).reshape(-1)
        savemat('D:\Study\Dissertation\Datasets\controlsig4.mat', {'yaw':yaws,'pitch':pitchs,'yaw_r':yaws_r,'pitch_r':pitchs_r})
        plt.figure(figsize=(10,5))
        #fig1
        plt.subplot(121)
        plt.plot(yaws,label="yaw_pre")
        plt.plot(yaws_r, label="yaw_rel")
        plt.xlabel("time")
        plt.ylabel("Yaw angle")
        plt.title('Yaw angle')
        #fig2
        plt.subplot(122)
        plt.plot(pitchs, label="pitch_pre")
        plt.plot(pitchs_r, label="pitch_rel")
        plt.xlabel("time")
        plt.ylabel("Pitch angle")
        plt.title("Pitch angle")
        plt.legend(loc="best")
        plt.savefig('angle.png')
        plt.show()
    if N == 2:
        yaw = 0
        yaws = []
        yaw_r = 0
        yaws_r = []
        with torch.no_grad():
            for i, data in enumerate(val_dataloader,0):
                    input = data[0].to(torch.float32)
                    labels = data[1].long()
                    input = input.unsqueeze(dim=1)
                    output = eegnetmodel(input)
                    _, predicted = torch.max(output.data, 1)
                    if predicted == 0:
                        yaw += -steplen
                        yaws.append([yaw]*1000)
                    elif predicted == 1:
                        yaw += steplen
                        yaws.append([yaw]*1000)
                    if labels == 0:
                        yaw_r += -steplen
                        yaws_r.append([yaw_r]*1000)
                    elif labels == 1:
                        yaw_r += steplen
                        yaws_r.append([yaw_r]*1000)
                    pred = predicted
                    y_true = labels.cpu()
                    y_pred = pred.float().cpu()
                    if len(predict)==0:
                        predict = y_pred.clone()
                        gt = y_true.clone()
                    else:
                        predict = torch.concat((predict,y_pred))
                        gt = torch.concat((gt,y_true))
        yaws = np.array(yaws).reshape(-1)
        yaws_r = np.array(yaws_r).reshape(-1)
        savemat('D:\Study\Dissertation\Datasets\controlsig2.mat', {'yaw':yaws,'yaw_r':yaws_r})
        plt.figure(figsize=(10,5))
        plt.plot(yaws, label="yaw_pre")
        plt.plot(yaws_r, label="yaw_rel")
        plt.xlabel("Time")
        plt.ylabel("Yaw angle")
        plt.title('Yaw angle')
        plt.legend(loc="best")
        plt.savefig('angle.png')
        plt.show()
    print(metrics.accuracy_score(gt,predict))
    print(metrics.classification_report(gt,predict))
    print(metrics.confusion_matrix(gt,predict))

if __name__ == "__main__":
    npzfile = np.load('D:\Study\Dissertation\Datasets\Alldata1000.npz') # Alldata1000,Data2type1000
    data = npzfile['data']
    labels = npzfile['labels']
    channelname = npzfile['channels']
    sbjnum = 4 # choose the 'num'th subject's data [1,2,3,4,5,6,7,8,9]
    data_sbj, label_sbj = load_sbjnum(sbjnum-1,data,labels)
    data_sbj = buttferfiter(data_sbj)
    x_train,x_test,y_train,y_test = train_test_split(data_sbj, label_sbj, 
                                                     train_size=0.7, random_state=12,stratify=label_sbj)
    x_train,y_train = cutdata(x_train,y_train,4)
    x_test,y_test = cutdata(x_test,y_test,4)
    x_train,x_test = normalization(x_train,x_test)
    # classnum(y_test,'label')
    train_data = EEGdata(x_train,y_train)
    test_data = EEGdata(x_test,y_test)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    # # draw all trails plot for one subject
    # for i in range(data_sbj.shape[0]):
    #     trail_plot(data_sbj[i],label_sbj1[i],channelname)

    # data_sbj,_ = normalization(data_sbj)
    # eegdata = EEGdata(data_sbj,label_sbj) # eegdata.data = data_sbj, eegdata.label = label_sbj
    # lenratio = 0.7 # ratio of training set (training set:validation set = 0.7:0.3)
    # len_train = int(len(eegdata)*lenratio)
    # torch.manual_seed(1)
    # train_data, test_data = random_split(eegdata,[len_train,len(eegdata)-len_train])
    # classnum(test_data)
    # train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    # val_loader = DataLoader(test_data, batch_size=4, shuffle=True)

    # when shuffle = true, the order of data in train_loader will change in every epoch
    # for epoch in range(3):
    #     for i, data in enumerate(train_loader):
    #         inputs, labels = data
    #     print("----------epoch-----------------------")

    lr = 0.001
    epochs = 600
    C = int(x_train.shape[1])
    T = int(x_train.shape[2])
    F = 8
    D = 2
    N = 4
    best_acc = 0
    eegnetmodel = EEGNeX(C, T, F, D, N)
    # constraints1=weightConstraint(1)
    # constraints2=weightConstraint(0.25)
    # eegnexmodel.block3._modules['0'].apply(constraints1) # limit the parameters of depthwise conv
    # eegnexmodel.classifier._modules['0'].apply(constraints2) # limit the parameters of fulllink 
    optimizer = optim.Adam(eegnetmodel.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    # loss, accuracy = train(epochs, optimizer, eegnetmodel, loss_fn, train_loader)
    # verify(eegnetmodel, train_loader)

    loss_list = []
    accuracy_list = []
    test_accuracy = []
    recall1= []
    recall2= []
    recall3 = []
    recall4 = []
    precision1 = []
    precision2 = []
    precision3 = []
    precision4 = []
    f1_1 = []
    f1_2 = []
    f1_3 = []
    f1_4 = []
    epoch_list = []
    traintime = 0
    for epoch in range(1,epochs+1):
        eegnetmodel.train()
        total = 0
        correct = 0
        Loss = 0
        start_time = time.time()
        for i, data in enumerate(train_loader,0):
            input = data[0].to(torch.float32)
            labels = data[1].long()
            input = input.unsqueeze(dim=1) #(batch,eegchannel,time)-->(batcch,channels,eegchannel,time)
            output = eegnetmodel(input)
            loss = loss_fn(output,labels)
            optimizer.zero_grad()
            # 注意在loss.backward()之前一定要将优化器的梯度置零，
            # 因为pytorch训练时优化器的梯度是累计的，在一批量迭代结束后，
            # 梯度并没有归零，如果不置零，则梯度会一直累加。
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            Loss += loss.item()
            # torch.clamp_max(,1)
            # torch.clamp_max(,0.25)
        end_time = time.time()
        train_time = end_time-start_time
        traintime = traintime+train_time
        print('训练时间：', train_time)
        accuracy = correct/total
        Loss = Loss/(total/labels.size(0))
        loss_list.append(Loss)
        accuracy_list.append(accuracy)
        print("Traning: Epoch=%d, Loss=%.4f, Accuracy=%.4f"%(epoch, Loss, accuracy))

        eegnetmodel.eval()
        with torch.no_grad():
            test_correct = 0
            total = 0
            tensor_concat_pre_label = []
            label_item = []
            target_num = torch.zeros((1, N)) 
            predict_num = torch.zeros((1, N))
            acc_num = torch.zeros((1, N))
            for i, data in enumerate(val_loader,0):
                input = data[0].to(torch.float32)
                labels = data[1].long()
                input = input.unsqueeze(dim=1)
                output = eegnetmodel(input)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                # 1 PR/RE/F1 报告
                pred = predicted
                y_true = labels.cpu()
                y_pred = pred.float().cpu()
                # print(y_pred,y_true)
                if len(tensor_concat_pre_label)==0:
                    tensor_concat_pre_label = y_pred.clone()
                    label_item = y_true.clone()
                else:
                    tensor_concat_pre_label = torch.concat((tensor_concat_pre_label,y_pred))
                    label_item = torch.concat((label_item,y_true))
                #3 每一类别的pr、re、f1图
                pre_mask = torch.zeros(output.size()).scatter_(1, predicted.view(-1, 1), 1.)
                predict_num += pre_mask.sum(0)  # 得到数据中每类的预测量

                tar_mask = torch.zeros(output.size()).scatter_(1, labels.view(-1, 1), 1.)
                target_num += tar_mask.sum(0)  # 得到数据中每类的数量
                
                acc_mask = pre_mask * tar_mask 
                acc_num += acc_mask.sum(0) # 得到各类别分类正确的样本数量
            recall = acc_num / target_num
            precision = acc_num / predict_num
            F1 = 2 * recall * precision / (recall + precision)
            recall  = recall.numpy()
            precision = precision.numpy()
            F1 = F1.numpy()
            # metrics.classification_report(label_item,tensor_concat_pre_label)
            # metrics.accuracy_score(label_item, tensor_concat_pre_label)
            # print(metrics.accuracy_score(label_item,tensor_concat_pre_label))
            # print(metrics.classification_report(label_item,tensor_concat_pre_label))
            # print(metrics.confusion_matrix(label_item,tensor_concat_pre_label))
            current_test_acc = test_correct/total
            test_accuracy.append(current_test_acc)
            print("Accuracy=%.4f"%(current_test_acc))
            if N == 4:
                recall_back1,recall_back2,recall_back3,recall_back4 = recall[:,0],recall[:,1],recall[:,2],recall[:,3]
                precision_back1,precision_back2,precision_back3,precision_back4 = precision[:,0],precision[:,1],precision[:,2],precision[:,3]
                F1_back1,F1_back2,F1_back3,F1_back4 = F1[:,0],F1[:,1],F1[:,2],F1[:,3]
                epoch_list.append(epoch)
                recall1.append(recall_back1)
                recall2.append(recall_back2)
                recall3.append(recall_back3)
                recall4.append(recall_back4)
                precision1.append(precision_back1)
                precision2.append(precision_back2)
                precision3.append(precision_back3)
                precision4.append(precision_back4)
                f1_1.append(F1_back1)  
                f1_2.append(F1_back2)
                f1_3.append(F1_back3)
                f1_4.append(F1_back4)
            elif N == 2:
                recall_back1,recall_back2 = recall[:,0],recall[:,1]
                precision_back1,precision_back2 = precision[:,0],precision[:,1]
                F1_back1,F1_back2 = F1[:,0],F1[:,1]
                epoch_list.append(epoch)
                recall1.append(recall_back1)
                recall2.append(recall_back2)
                precision1.append(precision_back1)
                precision2.append(precision_back2)
                f1_1.append(F1_back1)  
                f1_2.append(F1_back2)
            #accuracy = 100. * acc_num.sum(1) / target_num.sum(1)

            
        if current_test_acc > best_acc and epoch>epochs/2:
            torch.save(eegnetmodel.state_dict(), 'EEGNeXpara.pth')
            print(current_test_acc,best_acc)
            best_acc = current_test_acc
    print('总训练时间：', traintime)
    traintime = traintime/epochs
    print('平均训练时间：', traintime)
    # eegnetmodel.load_state_dict(best_model_wts)
    # if N == 4:
    #     plot_recall(epoch_list,recall1,recall2,recall3,recall4)
    #     plot_precision(epoch_list,precision1,precision2,precision3,precision4)
    #     plot_f1(epoch_list,f1_1,f1_2,f1_3,f1_4)
    # elif N == 2:
    #     plot_recall(epoch_list,recall1,recall2,N=N)
    #     plot_precision(epoch_list,precision1,precision2,N=N)
    #     plot_f1(epoch_list,f1_1,f1_2,N=N)
    show_plot(accuracy_list,loss_list,test_accuracy)
    DrawConfusionMatrix('EEGNeXpara.pth',C,T,F,D,N,val_loader)
    # getcontrolsig('EEGNetpara.pth',C,T,F,D,N,test_data)
    # torch.save(eegnexmodel,'D:\Study\Dissertation\Datasets\EEGNetmodel')
    # if current_test_acc > best_acc and epoch>Config.epochs/2:
    #     best_acc = current_test_acc
    #     torch.save(eegnexmodel.state_dict(), 'EEGNetpara.pth')

    # print(eegnexmodel.block3._modules['0'].state_dict().items()) # show depthwise conv parameters
    # plt.subplot(211)
    # plt.plot(loss)
    # plt.subplot(212)
    # plt.plot(accuracy)
    # plt.show()
    # plt.show()

    # model = torch.load('D:\Study\Dissertation\Datasets\EEGNetmodel')

    # with torch.no_grad():
    #     model.eval()
    #     for data in train_loader:
    #         input = data[0].to(torch.float32)
    #         labels = data[1].long()
    #         input = input.unsqueeze(dim=1)
    #         output = model(input)
    #         print(output.argmax(1),labels)