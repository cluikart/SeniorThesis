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
from functionLib import*
#import SingleNet_1
#from parser import *
                    
#print(drugResist1[0]) 
#print(drugResist2[0])
#print drugs



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




class SingleNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables."""
        
        super(SingleNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        #self.linear2 = torch.nn.Linear(H, H/2)
        #self.linear3 = torch.nn.Linear(H/2, H/4)
        self.linear4 = torch.nn.Linear(H, H//2)
        self.linear5 = torch.nn.Linear(H//2, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors."""
        
        #h_relu = self.linear1(x).clamp(min=0)
        #y_pred = self.linear2(h_relu)
        x = torch.sigmoid(self.linear1(x))
        #x = torch.sigmoid(self.linear2(x))
        #x = torch.sigmoid(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        y_pred = self.linear5(x)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
class SingleNet_Train:
    def __init__(self,samples,positions,H,out,drug, fullTarget, trainTarget, testTarget, fullInput, trainSet, testSet):
        self.N, self.D_in, self.H, self.D_out = samples, positions, H, out
        self.x = torch.reshape(fullInput[0],(-1,))
        self.yTrain = trainTarget
        self.yTest = testTarget
        self.y = fullTarget
        self.sXpXa = fullInput
        print("init Inputs")
        print(fullInput)
        print(self.sXpXa)
        self.samples = samples
        self.positions = positions
        self.lossDict = {}
        self.confusionMat = torch.zeros(500,2,2)
        self.confusionMatTest = torch.zeros(500,2,2)
        self.model = SingleNet(self.D_in, self.H, self.D_out)
        self.acc = np.zeros((1,500))
        self.prec = np.zeros((1,500))
        self.recall = np.zeros((1,500))
        self.fOut = np.zeros((1,500))
        self.accTest = np.zeros((1,500))
        self.precTest = np.zeros((1,500))
        self.recallTest = np.zeros((1,500))
        self.fOutTest = np.zeros((1,500))
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4)
        self.xs = torch.zeros([10, positions])
        self.ys = torch.zeros([10, 1])
        self.y_pred = []
        self.TPR_List = []
        self.FPR_List = []
        self.yList = []
        self.target = []
        self.yListTest = []
        self.targetTest = []
        self.y_temp = []
        self.drug = drug
        self.trainNum = int(math.floor(0.7 * self.samples))
        self.testNum = int(self.samples - self.trainNum)
        self.availableSamples = list(range(0,self.samples))
        self.testSet = testSet
        self.trainSet = trainSet
        self.trainTarget = trainTarget
        self.testTarget = testTarget
        self.wrongPred =[0,0,0,0,0,0,0,0,0,0]
        self.predList = []
        self.targetList = []
        for i in range(0,self.testNum):
            samp = random.randint(0, len(self.availableSamples)-1)
            #self.testSet.append(self.availableSamples[samp])
            #self.availableSamples.pop(samp)

        for i in range(0,self.trainNum):
            samp = random.randint(0, len(self.availableSamples)-1)
            #self.trainSamp.append(self.availableSamples[samp])
            #self.availableSamples.pop(samp)    
        #20,10


    def trainNet(self,epochs):
        print("train")
        print(self.trainSet)
        print("trainTarget")
        print(self.trainTarget)
        for w in range(epochs):
            self.yList = []
            self.target = []
            self.yListTest = []
            self.targetTest = []
            for t in range(self.samples//10):

                # Forward pass: Compute predicted y by passing x to the model
                self.xs = torch.zeros([10, self.positions])
                self.ys = torch.zeros([10, 1])
                #y[t*10:t*10+10:1,...]
                samp = 0
                positiveCount = 0
                while samp < 10:
                    rnd = random.randint(0,len(self.trainSet)-1)
                    self.xs[samp,] = torch.reshape(self.trainSet[rnd],(-1,))
                    self.ys[samp,] = torch.reshape(self.trainTarget[rnd, self.drug],(-1,))
                    if positiveCount < 5:
                        positiveCount += 1
                        samp += 1
                    elif self.ys[samp,0].item != 1:
                        samp+= 1
               
                    # 1985 for all, 1916 for just type B
                    #xs[i,] = torch.reshape(sXpXa[i+t*10,...,...],(-1,))

                
                #conMat2(self.y_pred,self.ys,w, self.confusionMat)

                self.y_pred = self.model(self.xs)
                # Compute and print loss
                self.loss = self.criterion(self.y_pred, self.ys.type(torch.FloatTensor))
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                
                
                #for i in range(0, 10):
                #    if self.y_pred[i] < 0.7:
                #        self.y_pred[i] = 0
                #    else:
                #        self.y_pred[i] = 1
                
                
                self.y_temp = self.y_pred.detach().numpy().tolist()
                self.yList = self.yList + [item for sublist in self.y_temp for item in sublist]

                self.ys_temp = self.ys.numpy().tolist()
                self.target = self.target + [item for sublist in self.ys_temp for item in sublist]

               
                #Randomly select batch set, but balance negative and positive samples
                samp = 0
                posCnt = 0 
                while samp < 10:
                    rnd = random.randint(0,len(self.testSet)-1)
                    self.xs[samp,] = torch.reshape(self.testSet[rnd],(-1,))
                    self.ys[samp,] = torch.reshape(self.testTarget[rnd, self.drug],(-1,))
                    if samp < 6:
                        samp += 1
                        posCnt += 1
                    elif self.ys[samp,0].item() != 1:
                        samp += 1
                        
                    # 1985 for all, 1916 for just type B
                    #xs[i,] = torch.reshape(sXpXa[i+t*10,...,...],(-1,))

                self.y_pred = self.model(self.xs)
                conMat2(self.y_pred,self.ys,w, self.confusionMatTest)
                """
                for i in range(0, 10):
                    if self.y_pred[i] < 0.7:
                        self.y_pred[i] = 0
                    else:
                        self.y_pred[i] = 1
                """
                self.y_temp = self.y_pred.detach().numpy().tolist()
                self.yListTest = self.yListTest + [item for sublist in self.y_temp for item in sublist]

                self.ys_temp = self.ys.numpy().tolist()
                self.targetTest = self.targetTest + [item for sublist in self.ys_temp for item in sublist]

                # Zero gradients, perform a backward pass, and update the weights.

            #statRec(w,self.acc,self.prec,self.recall,self.fOut,self.confusionMat)
            #self.acc[0,w] = metrics.accuracy_score(self.target, self.yList)
            #self.prec[0,w] = metrics.precision_score(self.target, self.yList)
            #self.recall[0,w] = metrics.recall_score(self.target, self.yList)
            #e = metrics.confusion_matrix(self.target, self.yList).ravel()
            #if(len(e) < 2):
            #    self.fOut[0,w] = 0
            #else:
            #    fp, tn, fn, tp = e
            #    self.fOut[0,w] = fp/(fp+tn+fn+tp)
            
            """
            self.accTest[0,w] = metrics.accuracy_score(self.targetTest, self.yListTest)
            self.precTest[0,w] = metrics.precision_score(self.targetTest, self.yListTest)
            self.recallTest[0,w] = metrics.recall_score(self.targetTest, self.yListTest)
            e = metrics.confusion_matrix(self.targetTest, self.yListTest).ravel()
            if len(e) < 2:
                self.fOutTest[0,w] = 0
            else:
                fp, tn, fn, tp = e
                self.fOutTest[0,w] = fp/(fp+tn+fn+tp)
            """
            print("TargetTest")
            print(self.targetTest)
            print("PedictedTest")
            print(self.yListTest)
            print("TargetTrain")
            print(self.target)
            print("PredictedTrain")
            print(self.yList)
            fpr, tpr, thres = metrics.roc_curve(self.targetTest, self.yListTest)
            print(fpr)
            print(tpr)
            roc_auc = metrics.auc(fpr,tpr)
            print('ROC fold %d (AUC = %0.2f)' % (0, roc_auc))
            plt.plot(fpr, tpr, lw=1, color =(0.0, 0.9, 0.0, w/10),label='ROC fold %d (AUC = %0.2f)' % (0, roc_auc))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
        drugs = ["3Tc", "ABC", "AZT","D4T","DDI","TDF", "EFV", "NVP", "ETR", "RPV"]
        plt.savefig("/data/hibbslab/cluikart/exp50_post/aucTest_"+str(drugs[self.drug])+ ".png")
        plt.close()
            
            

        #acc = accuracy(lossDict)
        #for i in range(0,len(drugs)):
            #print (drugs[i], acc[i])

        #print(self.confusionMat)
        """
        t = np.arange(0., 10., 1)
        plt.plot(t, self.acc[0,0:10], 'r--', t, self.prec[0,0:10], 'bs', t, self.recall[0,0:10], 'g^', t, self.fOut[0,0:10], "m+")
        plt.savefig("/data/hibbslab/cluikart/train_"+str(self.drug)+ ".png")
        plt.close()

        t = np.arange(0., 10., 1)
        plt.plot(t, self.accTest[0,0:10], 'r--', t, self.precTest[0,0:10], 'bs', t, self.recallTest[0,0:10], 'g^', t, self.fOutTest[0,0:10], "m+")
        plt.savefig("/data/hibbslab/cluikart/test_"+ str(self.drug)+ ".png")
        plt.close()
        #plt.show()
        torch.save(self.confusionMat, "conMat_"+ str(self.drug) + ".pth")
        torch.save(self.confusionMatTest, "conMatTest_"+ str(self.drug) + ".pth")
        """

    
    def testNet(self,epochs):
        
        self.confusionMat = torch.zeros(500,2,2)
        for w in range(epochs):
            for t in range(self.samples//10):

                # Forward pass: Compute predicted y by passing x to the model
                self.xs = torch.zeros([10, self.positions])
                self.ys = torch.zeros([10, 10])
                #y[t*10:t*10+10:1,...]

                for i in range(0,10):
                    rnd = random.randint(0,len(self.testSet)-1)
                    self.xs[i,] = self.testSet[rnd]
                    self.ys[i,] = self.yTest[rnd,self.drug]
                    # 1985 for all, 1916 for just type B
                    #xs[i,] = torch.reshape(sXpXa[i+t*10,...,...],(-1,))

                self.y_pred = self.model(self.xs)
                #metrics.confusion_matrix(self.ys, self.y_pred)
                #conMat(self.y_pred,self.ys,w, self.confusionMat)
                
                #for entry in range(0,len(self.ys)):
                    #for drug in range(0,len(self.ys[1,])):
                        #if(abs(self.ys[entry,drug] - self.y_pred[entry,drug]) > .5):
                            #self.wrongPred[drug] += 1
                        #elif(abs(self.ys[entry,drug] - self.y_pred[entry,drug]) <= .5):
                                #a =1

                # Compute and print loss
                #self.loss = self.criterion(self.y_pred, self.ys.type(torch.FloatTensor))
                

                # Zero gradients, perform a backward pass, and update the weights.
                #self.optimizer.zero_grad()
                #self.loss.backward()
                #self.optimizer.step()

            statRec(w,self.acc,self.prec,self.recall,self.fOut,self.confusionMat)

        #acc = accuracy(lossDict)
        drugList = ["3TC", "ABC", "AZT","D4T","DDI","TDF", "EFV", "NVP", "ETR", "RPV"]
        #for i in range(0,len(drugs)):
            #print (drugs[i], acc[i])

        #print(self.confusionMat)

        t = np.arange(0., 10., 1)
        plt.plot(t, self.acc[0,0:10], 'r--', t, self.prec[0,0:10], 'bs', t, self.recall[0,0:10], 'g^', t, self.fOut[0,0:10], "m+")
       # plt.savefig("/data/hibbslab/cluikart/test_"+ str(self.drug)+ ".png")
        plt.close()
        for item in range(0,len(self.wrongPred)):
            print (drugList[item] + " " + str((float(self.wrongPred[item])/(len(self.testSet)*epochs))*100) + "%")
        #plt.show()


    def buildROC(self):
        

        i = 0

        for t in range(self.samples//10):

            #w = int(np.asscalar(r))
            # Forward pass: Compute predicted y by passing x to the model
            self.xs = torch.zeros([10, self.positions])
            self.ys = torch.zeros([10, 1])
            #y[t*10:t*10+10:1,...]

            for i in range(0,10):
                rnd = random.randint(1,1915)
                self.xs[i,] = torch.reshape(self.sXpXa[rnd,...],(-1,))
                self.ys[i,] = torch.reshape(self.y[rnd,self.drug],(-1,))
                # 1985 for all, 1916 for just type B


            self.y_pred = self.model(self.xs)
            self.y_temp = self.y_pred.detach().numpy().tolist()
            self.yList = self.yList + [item for sublist in self.y_temp for item in sublist]

            self.ys_temp = self.ys.numpy().tolist()
            self.target = self.target + [item for sublist in self.ys_temp for item in sublist]




            # Compute and print loss
            self.loss = self.criterion(self.y_pred, self.ys.type(torch.FloatTensor))

            i+=1


        #plt.plot(FPR_List, TPR_List)

        fpr, tpr, thres = metrics.roc_curve(self.target, self.yList)
        roc_auc = metrics.auc(fpr,tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)

        plt.savefig("/data/hibbslab/cluikart/auc_"+str(self.drug)+ ".png")
        plt.close()
        #plt.show()





