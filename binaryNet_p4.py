import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
#from parser import *
from functionLib import*

inFile = open("RT_PhenoSense_DataSet.txt")

#Calculate the number of samples for training and testing
#Then generate lists of random indexis into the whole sample. Do this for trainig and test set.
#Steps: 1. Make a list of availble indexes of the whole sample set
#       2. Randomly select indexes of the available set, add to test or train set
#       3. Remove index from available pool
  

class My_Loss(torch.nn.Module):
    
    def __init__(self):
        super(My_Loss,self).__init__()
        
    def forward(self,x,y):
        xy = torch.dot(x.view(-1),y.view(-1))
        Sxy = torch.sum(xy)
        Sx = torch.sum(x)
        Sy = torch.sum(y)
        SxSy = Sx*Sy
        x_sq = torch.dot(x.view(-1),x.view(-1))
        Sx_sq = torch.sum(x_sq)
        y_sq = torch.dot(y.view(-1),y.view(-1))
        Sy_sq = torch.sum(y_sq)
        sq_Sx= Sx*Sx
        sq_Sy = Sy*Sy
        numerator = (100*Sxy - Sx*Sy)**2
        denom = (100*Sx_sq - sq_Sx)*(100*Sy_sq - sq_Sy) #fix * to - later
        r_sq = numerator / denom
        #y_shape = y.size()[1]
        #x_added_dim = x.unsqueeze(1)
        #x_stacked_along_dimension1 = x_added_dim.repeat(1,NUM_WORDS,1)
        #diff = torch.sum((y - x_stacked_along_dimension1)**2,2)
        #totloss = torch.sum(torch.sum(torch.sum(diff)))
        return 1-r_sq




class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables."""
        
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H//2)
        self.linear3 = torch.nn.Linear(H//2, H//4)
        self.linear4 = torch.nn.Linear(H//4, H//8)
        self.linear5 = torch.nn.Linear(H//8, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors."""
        
        #h_relu = self.linear1(x).clamp(min=0)
        #y_pred = self.linear2(h_relu)
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        y_pred = self.linear5(x)
        return y_pred


def conMat(y_pred, ys,w,cMat):
    for i in range(0,10):
        for j in range(0,10):
            if(ys[i,j] == 1):
                if(y_pred[i,j] > 0.7):
                    cMat[w,0,0] += 1
                else:
                    cMat[w,0,1] += 1
            elif(ys[i,j] == 0):
                if(y_pred[i,j] > 0.7):
                    cMat[w,1,0] += 1
                else:
                    cMat[w,1,1] += 1
    return cMat


def lossRecorder(x, y, i, lossDict):
    tmp = torch.zeros(1,10)
    diff = torch.abs(y - x)
    for j in range(0,10):
        tmp[0,j] = torch.sum(diff[...,j])
    lossDict[i] = tmp

"""def accuracy(lossDict):
    acc = [0,0,0,0,0,0,0,0,0,0,0]
    for j in range(0, 10):

        for i in range(0, len(lossDict)):

            acc[j] += lossDict[i][0,j]

        acc[j] = acc[j]/len(lossDict)
    return acc
      """  

confusionMat = torch.zeros(500,2,2)    
confusionMatTest = torch.zeros(500,2,2)
    
def runNet(sXpXa, drugResist2, samples, positions,epoch):  
    
    trainNum = int(math.floor(0.7 * samples))
    testNum = int(samples - trainNum)
    availableSamples = list(range(0,samples))
    testSamp = []
    trainSamp = []
    for i in range(0,testNum):
        samp = random.randint(0, len(availableSamples)-1)
        testSamp.append(availableSamples[samp])
        availableSamples.pop(samp)

    for i in range(0,trainNum):
        samp = random.randint(0, len(availableSamples)-1)
        trainSamp.append(availableSamples[samp])
        availableSamples.pop(samp)  
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = samples, positions, 100, 10

    # Create random Tensors to hold inputs and outputs
    x = torch.reshape(sXpXa[0],(-1,))
    y = drugResist2



    #x = torch.randn(N, D_in)
    #y = torch.randn(N, D_out)

    lossDict = {}

    global confusionMat
    global confusionMatTest
    
    # Construct our model by instantiating the class defined above
    model = TwoLayerNet(D_in, H, D_out)

    acc = np.zeros((1,500))
    prec = np.zeros((1,500))
    recall = np.zeros((1,500))
    fOut = np.zeros((1,500))


    accTest = np.zeros((1,500))
    precTest = np.zeros((1,500))
    recallTest = np.zeros((1,500))
    fOutTest = np.zeros((1,500))
    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    #criterion = torch.nn.L1Loss()
    criterion = nn.MultiLabelMarginLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    #####Train
    #####
    for w in range(epoch):
        for t in range(samples//10):

            # Forward pass: Compute predicted y by passing x to the model
            xs = torch.zeros([10, positions])
            ys = torch.zeros([10, 10])
            #y[t*10:t*10+10:1,...]

            #Randomly select a sample, recording negative and positive labels in Target
            #If sample exceeds 50/50 balance ratio, choose another sample
            #Only select samples which produce 50/50 negative positive balance
            samp = 0
            positiveCount = 0
            negativeCount = 0
            while samp < 10:
                
                while positiveCount < 50 and negativeCount < 50:
                    rnd = random.randint(0,len(trainSamp)-1)
                    xs[samp,] = torch.reshape(sXpXa[trainSamp[rnd],...],(-1,))
                    ys[samp,] = torch.reshape(y[trainSamp[rnd],...],(-1,))
                    for entry in ys[samp,]:
                            if entry == 1:
                                positiveCount += 1   
                            else:
                                negativeCount += 1
                    samp += 1            
                    
                    diff = positiveCount - negativeCount
                    diffMatch = false
                    s = 0
                    pCount = 0
                    nCount = 0
                    sample = torch.reshape(y[trainSamp[s],...],(-1,))
                    while not diffMatch and s < len(y):
                        for entry in sample:
                            if entry == 1:
                                pCount += 1   
                            else:
                                nCount += 1
                        if nCount-pCount == diff:
                            diffMatch = true
                            positiveCount += pCount
                            negativeCount += nCount
                        else:
                            s += 1
                            sample = torch.reshape(y[trainSamp[s],...],(-1,))
                    xs[samp,] = torch.reshape(sXpXa[trainSamp[s],...],(-1,))
                    ys[samp,] = sample      
                    samp += 1          
                            
                      
                                    
                        
                # 1985 for all, 1916 for just type B
                #xs[i,] = torch.reshape(sXpXa[i+t*10,...,...],(-1,))

            y_pred = model(xs)
            lossRecorder(y_pred,ys,t, lossDict)
            conMat(y_pred,ys,w, confusionMat)

            for i in range(0,10):
                rnd = random.randint(0,len(testSamp)-1)
                #rnd = random.randint(238,1676)
                xs[i,] = torch.reshape(sXpXa[testSamp[rnd],...],(-1,))
                ys[i,] = torch.reshape(y[testSamp[rnd],...],(-1,))
 
            y_pred = model(xs)
            conMat(y_pred,ys,w,confusionMatTest)
            # Compute and print loss
            loss = criterion(y_pred, ys.type(torch.LongTensor))
            

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc[0,w] = (confusionMat[w,0,0] + confusionMat[w,1,1])/(confusionMat[w,0,0] + confusionMat[w,1,0] + confusionMat[w,0,1] + confusionMat[w,1,1])
        prec[0,w] = confusionMat[w,0,0] / (confusionMat[w,0,0] + confusionMat[w,1,0])

        recall[0,w] = confusionMat[w,0,0] / (confusionMat[w,0,0] + confusionMat[w,0,1])

        fOut[0,w] = confusionMat[w,1,0] / (confusionMat[w,1,0] + confusionMat[w,1,1])

        accTest[0,w] = (confusionMatTest[w,0,0] + confusionMatTest[w,1,1])/(confusionMatTest[w,0,0] + confusionMatTest[w,1,0] + confusionMatTest[w,0,1] + confusionMatTest[w,1,1])
        precTest[0,w] = confusionMatTest[w,0,0] / (confusionMatTest[w,0,0] + confusionMatTest[w,1,0])

        recallTest[0,w] = confusionMatTest[w,0,0] / (confusionMatTest[w,0,0] + confusionMatTest[w,0,1])

        fOutTest[0,w] = confusionMatTest[w,1,0] / (confusionMatTest[w,1,0] + confusionMatTest[w,1,1])
    #acc = accuracy(lossDict)
    drugList = ["3TC","ABC","AZT","D4T","DDI","TDF","EFV","NVP","ETR","RPV"]
    #for i in range(0,len(drugs)):
        #print (drugs[i], acc[i])

    #print(confusionMat)

    accuracy = (confusionMat[...,0,0] + confusionMat[...,1,1])/torch.sum(confusionMat)

    precision = confusionMat[...,0,0] / (confusionMat[...,0,0] + confusionMat[...,1,0])

    Recall = confusionMat[...,0,0] / (confusionMat[...,0,0] + confusionMat[...,0,1])

    fallOut = confusionMat[...,1,0] / (confusionMat[...,1,0] + confusionMat[...,1,1])
    #print("accuracy: " , accuracy)
    #print("precision: " , precision)
    #print("Recall: " , Recall)
    #print("fallOut: " , fallOut)

    t = np.arange(0., 10., 1)
    plt.plot(t, acc[0,0:10], 'r--', t, prec[0,0:10], 'bs', t, recall[0,0:10], 'g^', t, fOut[0,0:10], "m+")
    plt.savefig("/data/hibbslab/cluikart/MultiDrug_train.pdf")
    plt.close()

    plt.plot(t, accTest[0,0:10], 'r--', t, precTest[0,0:10], 'bs', t, recallTest[0,0:10], 'g^', t, fOutTest[0,0:10], "m+")
    plt.savefig("/data/hibbslab/cluikart/MultiDrug_test.pdf")
    plt.close()
    #plt.show()
    torch.save(confusionMat, 'conMat.pth')
    torch.save(confusionMatTest, 'conMatTest.pth')

    confusionMat = torch.zeros(500,2,2)
    wrongPred =[0,0,0,0,0,0,0,0,0,0]
    #####Test
    for w in range(epoch):
        for t in range(samples//10):

            # Forward pass: Compute predicted y by passing x to the model
            xs = torch.zeros([10, positions])
            ys = torch.zeros([10, 10])
            #y[t*10:t*10+10:1,...]

            
            for i in range(0,10):
                rnd = random.randint(0,len(testSamp)-1)
                #rnd = random.randint(238,1676)
                xs[i,] = torch.reshape(sXpXa[testSamp[rnd],...],(-1,))
                ys[i,] = torch.reshape(y[testSamp[rnd],...],(-1,))
                # 1985 for all, 1916 for just type B
                #xs[i,] = torch.reshape(sXpXa[i+t*10,...,...],(-1,))

            y_pred = model(xs)
            lossRecorder(y_pred,ys,t, lossDict)
            #conMat(y_pred,ys,w)
            
            for entry in range(0,len(ys)):
                for drug in range(0,len(ys[1,])):
                    if(abs(ys[entry,drug] - y_pred[entry,drug]) > .5):
                        wrongPred[drug] += 1
                    elif(abs(ys[entry,drug] - y_pred[entry,drug]) <= .5):
                        a = 1
                        #Do Nothing

            # Compute and print loss
            loss = criterion(y_pred, ys.type(torch.LongTensor))
            if(w > 8):
                a=2
                #print(t, loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc[0,w] = (confusionMat[w,0,0] + confusionMat[w,1,1])/(confusionMat[w,0,0] + confusionMat[w,1,0] + confusionMat[w,0,1] + confusionMat[w,1,1])
        prec[0,w] = confusionMat[w,0,0] / (confusionMat[w,0,0] + confusionMat[w,1,0])

        recall[0,w] = confusionMat[w,0,0] / (confusionMat[w,0,0] + confusionMat[w,0,1])

        fOut[0,w] = confusionMat[w,1,0] / (confusionMat[w,1,0] + confusionMat[w,1,1])

    #acc = accuracy(lossDict)
    drugs = ["3Tc", "ABC", "AZT","D4T","DDI","TDF", "EFV", "NVP", "ETR", "RPV"]
    #for i in range(0,len(drugs)):
        #print (drugs[i], acc[i])

    print(confusionMat)

    accuracy = (confusionMat[...,0,0] + confusionMat[...,1,1])/torch.sum(confusionMat)

    precision = confusionMat[...,0,0] / (confusionMat[...,0,0] + confusionMat[...,1,0])

    Recall = confusionMat[...,0,0] / (confusionMat[...,0,0] + confusionMat[...,0,1])

    fallOut = confusionMat[...,1,0] / (confusionMat[...,1,0] + confusionMat[...,1,1])
    #print("accuracy: " , accuracy)
    #print("precision: " , precision)
   # print("Recall: " , Recall)
    #print("fallOut: " , fallOut)

    t = np.arange(0., 10., 1)
    plt.plot(t, acc[0,0:10], 'r--', t, prec[0,0:10], 'bs', t, recall[0,0:10], 'g^', t, fOut[0,0:10], "m+")
    #plt.savefig("/data/hibbslab/cluikart/MultiDrug_test.pdf")
    plt.close()
    
    ret = [float(x)/(len(drugResist2)*epoch) for x in wrongPred]
    for item in range(0,len(wrongPred)):
        print( drugList[item] + " " + str((float(wrongPred[item])/(len(drugResist2)*epoch))*100) + "%")
    #plt.show()







    TPR_List = []
    FPR_List = []
    yList = []
    target = []

    i = 0

    confusionMat = torch.zeros(500,2,2)
    
    for t in range(samples//10):

        #w = int(np.asscalar(r))
        # Forward pass: Compute predicted y by passing x to the model
        xs = torch.zeros([10, positions])
        ys = torch.zeros([10, 10])
        #y[t*10:t*10+10:1,...]

        for i in range(0,10):
            rnd = random.randint(1,samples -1)
            xs[i,] = torch.reshape(sXpXa[rnd,...],(-1,))
            ys[i,] = torch.reshape(y[rnd,...],(-1,))
            # 1985 for all, 1916 for just type B


        y_pred = model(xs)
        y_temp = y_pred.detach().numpy().tolist()
        yList = yList + [item for sublist in y_temp for item in sublist]

        ys_temp = ys.numpy().tolist()
        target = target + [item for sublist in ys_temp for item in sublist]

        lossRecorder(y_pred,ys,t, lossDict)
        conMat2(y_pred,ys,w, confusionMat)


        # Compute and print loss
        loss = criterion(y_pred, ys.type(torch.LongTensor))
        
        i+=1

        # Zero gradients, perform a backward pass, and update the weights.
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()

    TPR = (confusionMat[w,0,0] / (confusionMat[w,0,0] + confusionMat[w,0,1]))

    TPR_List.append(TPR.item())

    FPR = (confusionMat[w,1,0] / (confusionMat[w,1,0] + confusionMat[w,1,1]))

    FPR_List.append(FPR.item())

    #plt.plot(FPR_List, TPR_List)

    fpr, tpr, thres = metrics.roc_curve(target, yList)
    roc_auc = metrics.auc(fpr,tpr)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)

    plt.savefig("/data/hibbslab/cluikart/MultiDrug_auc.pdf")
    plt.close()
    #plt.show()

