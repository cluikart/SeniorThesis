from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import zip_longest
from jsonParser import*
import subprocess
import SingleNet_1_Cluster as sn
import sys
import argparse
import binaryNet_p4 as bn

text ="Description goes here"

parser = argparse.ArgumentParser(description = text)
parser.add_argument("-E", "--expand", help="set output expand", type=int)
parser.add_argument("-T1", "--target1", action="store_true")
parser.add_argument("-T2", "--target2", action="store_true")
parser.add_argument("-sf", "--small_files", action="store_true")
parser.add_argument("-sb", "--submit", action="store_true")
parser.add_argument("-in", "--inputTensor", action="store_true")
parser.add_argument("-stat", "--statistics", action="store_true")
parser.add_argument("-tr", "--train", action="store_true")
parser.add_argument("-a", "--All", help="set output all", type=int)



args = parser.parse_args();


def grouper(n, iterable, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)



def printNuc(nuc):
    for n in nuc:
        print(str([x for x in n if x != '-' and x != '.']))




inFile = open("RT_PhenoSense_DataSet.txt")


nucleotides = []
drugFold = []
drugFoldMatch = []
seqID = []
i = 0
allLines = inFile.readlines()
for line in allLines:
    items = line.split("\t")
    
    if(items[2] == "B"):
        i += 1
        drugFold.append(items[7:27:2])
        drugFoldMatch.append(items[8:28:2])
        nucleotides.append([items[x] for x in range(28,587)])
        #print(items[0])
        seqID.append(items[0])
        #print(i)
    


#AA = "GALMFWKQESPVICYHRNDT"
AA = "ARNDBCEQZGHILKMFPSTWYV"

drugList = ["3TC","ABC","AZT","D4T","DDI","TDF","EFV","NVP","ETR","RPV"]

RT_Consensus = "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDKDFRKYTAFTIPSINNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGFTTPDKKHQKEPPFLWMGYELHPDKWTVQPIVLPEKDSWTVNDIQKLVGKLNWASQIYAGIKVKQLCKLLRGTKALTEVIPLTEEAELELAENREILKEPVHGVYYDPSKDLIAEIQKQGQGQWTYQIYQEPFKNLKGKYARMRGAHTNDVKQLTEAVQKIATESIVIWGKTPKFKLPIQKETWEAWWTEYWQATWIPEWEFVNTPPLVKLWYQLEKEPIVGAETFYVDGAANRETKLGKAGYVTDRGRQKVVSLTDTTNQKTELQAIHLALQDSGLEVNIVTDSQYALGIIQAQPDKSESELVSQIIEQLIKKEKVYLAWVPAHKGIGGNEQVDKLVSAGIRKVL"

"""for sequence in nucleotides:
    for amino in range(0,len(sequence)):
        if(sequence[amino] == '-'):
            sequence[amino] = RT_Consensus[amino]

print nucleotides[1]"""
samples = len(nucleotides)
print(samples)
positions = len(RT_Consensus)
#print(positions)
aminos = len(AA)
a = 0

testSet = []
testTarget = []
trainSet = []
trainTarget = []

for i in range(0,len(nucleotides)):
    rnd = random.randint(0,10)
    if( rnd < 3):
        testSet.append(nucleotides[i])
        testTarget.append(drugFold[i])
    else:
        trainSet.append(nucleotides[i])
        trainTarget.append(drugFold[i])


# Sample x position x Amino

sample = 0   
combo = 1
ID = seqID[0]
if args.All:
    args.expand = args.All

print ("Expanding Sequences")
print (args.expand)
while(sample < len(seqID)):
    
    for pos in range(0,positions):
        if(len(nucleotides[sample][pos]) > 1):
            combo = combo*len(nucleotides[sample][pos])
        elif(nucleotides[sample][pos] == 'X'):
            combo = combo*aminos
    #Change This Later        
    #cutOff = 300
    if(1 < combo and combo < args.expand):
        pos = 0
        expanded = False
        while(pos < positions and not expanded):
            if(len(nucleotides[sample][pos]) > 1):
                #print("Variations at position: " + str(len(nucleotides[sample][pos])))
                for letter in nucleotides[sample][pos]:
                    #print(sample)
                    a = nucleotides[sample][:]
                    a[pos] = letter
                    a.append(len(seqID))
                    nucleotides.append(a) 
                    a = seqID[sample]
                    seqID.append(a)
                    b = drugFoldMatch[sample]
                    drugFoldMatch.append(b)
                    c = drugFold[sample]
                    drugFold.append(c)
                seqID.pop(sample)    
                nucleotides.pop(sample)    
                drugFoldMatch.pop(sample)
                drugFold.pop(sample)
                sample -= 1
               
                    
                expanded = True
                
            elif(nucleotides[sample][pos] == 'X'):
                #print("Variations at position: " + str(len(AA)))
                for amino in AA:
                    a = nucleotides[sample][:]
                    a[pos] = amino
                    a.append(len(seqID))
                    nucleotides.append(a)
                    a = seqID[sample]
                    seqID.append(a)
                    b = drugFoldMatch[sample]
                    drugFoldMatch.append(b)
                    c = drugFold[sample]
                    drugFold.append(c)
                nucleotides.pop(sample)
                seqID.pop(sample)
                drugFoldMatch.pop(sample)
                drugFold.pop(sample)
                expanded = True
                sample -= 1     
            pos += 1
        #print("Seq ID: " + str(seqID))
    combo = 1
    sample += 1

#Recalculate sample size after expansion
#print("Count of first Entry Expanded")
#print(seqID.count(ID))
samples = len(nucleotides)
print("Number of Samples Post Expansion")
print(samples)
sXpXa = torch.zeros([samples,positions])
dbInput = nucleotides[:]

#inpt = raw_input("Build Input Tensor: y/n   ")
if args.inputTensor or args.All:

    #inpt = raw_input("Build small_files: y/n    ")

    if args.small_files or args.All:
        f = open("/data/hibbslab/cluikart/HivDB.fasta", "w+") 

    print("Building input Tensor")
    for sample in range(0,samples):
        if args.small_files or args.All:
            f.write(">" + str(sample))
            f.write('\n')
        for pos in range(0,positions):
            if(nucleotides[sample][pos] == '-'):
                dbInput[sample][pos] = RT_Consensus[pos]
                a = 1
            else:
                for amino in range(0,aminos):
                    if(nucleotides[sample][pos] == AA[amino]):
                        sXpXa[sample,pos] = amino+1
            if args.small_files or args.All:            
                f.write(dbInput[sample][pos]) 
        if args.small_files or args.All:
            f.write('\n')
    if args.small_files or args.All:
        f.close()

    #NOTE: File is divided into units of 1000 due to limits of Emboss and Stanford Web Programs    
    n = 500
    if args.small_files or args.All:
        print ("Building small_files")
        with open('/data/hibbslab/cluikart/HivDB.fasta') as f:
            for i, g in enumerate(grouper(n, f, fillvalue=''), 1):
                with open('/data/hibbslab/cluikart/small_file_{0}'.format(i * n), 'w') as fout:
                    fout.writelines(g)
    
    #inpt = raw_input("Submit small_files to Standford HIVdb: y/n    ")
    if args.submit or args.All:
        #NOTE: Sequences must first be backtranslated from AA's to NA's using Emboss Web App
        print( "Submitting Files (this may take a while)")
        subprocess.call(["chmod", "u+x", "reverseTranslate_cluster.sh"])
        subprocess.call(["./reverseTranslate_cluster.sh"])
        subprocess.call(["chmod", "u+x", "runHIVdb_Cluster.sh"])
        subprocess.call(["./runHIVdb_Cluster.sh"])


#print(drugFold[1])
drugs = len(drugFold[0])
#inpt = raw_input("Build Target Tensor #1: y/n   ")
if args.target1 or args.All:
    
    print("Building target Tensor #1")
    drugResist1 = torch.zeros([samples, 10])
    for sample in range(1,samples):
            #print sample
            for drug in range(0,drugs):
                if(drugFold[sample][drug] == 'NA' or float(drugFold[sample][drug]) <= 1):
                    drugResist1[sample][drug] = 0
                else:
                    drugResist1[sample][drug] = 1

 
#inpt = raw_input("Build Target Tensor #2: y/n   ")
if args.target2 or args.All:                    
    print("Building target Tensor #2")    
    drugResist2 = torch.zeros([samples, 10])
    for sample in range(1,samples):
            for drug in range(0,drugs):
                if(drugFoldMatch[sample][drug] == 'NA' or drugFoldMatch[sample][drug] == '<'):
                    drugResist2[sample][drug] = 0
                elif(drugFoldMatch[sample][drug] == '>' or drugFoldMatch[sample][drug] == '='):
                    drugResist2[sample][drug] = 1
                    
count = 0
#inpt = raw_input("Calculate Stats On HIVdb Program Analysis: y/n    ")
if args.statistics or args.All:                    
    print("Calculating Misclassification Rates")                      
    readJSON()
    wrongPred =[0,0,0,0,0,0,0,0,0,0]
    for sample in range(0,len(drugResist2)):
        for drug in range(1,11):
            if drugResist2[sample][drug-1] != data[sample][drug][1]:
                wrongPred[drug-1] += 1
            else:
                count += 1
                #print str(drugResist2[sample][drug-1]) + str(data[sample][drug][1])
    ret = [x/len(drugResist2) for x in wrongPred]

    for item in range(0,len(wrongPred)):
        print( drugList[item] + " " + str((wrongPred[item]/len(drugResist2))*100) + "%")
     




#Run the 11 Networks
if args.train or args.All:
    
    bn.runNet(sXpXa, drugResist2, samples, positions, 10)
    
    
    SingleNet_1 = sn.SingleNet_Train(samples,positions,20,10,0,drugResist2, sXpXa)
    SingleNet_1.trainNet(10)
    SingleNet_1.testNet(10)
    SingleNet_1.buildROC()

    SingleNet_2 = sn.SingleNet_Train(samples,positions,20,10,1,drugResist2, sXpXa)
    SingleNet_2.trainNet(10)
    SingleNet_2.testNet(10)
    SingleNet_2.buildROC()

    SingleNet_3 = sn.SingleNet_Train(samples,positions,20,10,2,drugResist2, sXpXa)
    SingleNet_3.trainNet(10)
    SingleNet_3.testNet(10)
    SingleNet_3.buildROC()

    SingleNet_4 = sn.SingleNet_Train(samples,positions,20,10,3,drugResist2, sXpXa)
    SingleNet_4.trainNet(10)
    SingleNet_4.testNet(10)
    SingleNet_4.buildROC()

    SingleNet_5 = sn.SingleNet_Train(samples,positions,20,10,4,drugResist2, sXpXa)
    SingleNet_5.trainNet(10)
    SingleNet_5.testNet(10)
    SingleNet_5.buildROC()

    SingleNet_6 = sn.SingleNet_Train(samples,positions,20,10,5,drugResist2, sXpXa)
    SingleNet_6.trainNet(10)
    SingleNet_6.testNet(10)
    SingleNet_6.buildROC()

    SingleNet_7 = sn.SingleNet_Train(samples,positions,20,10,6,drugResist2, sXpXa)
    SingleNet_7.trainNet(10)
    SingleNet_7.testNet(10)
    SingleNet_7.buildROC()

    SingleNet_8 = sn.SingleNet_Train(samples,positions,20,10,7,drugResist2, sXpXa)
    SingleNet_8.trainNet(10)
    SingleNet_8.testNet(10)
    SingleNet_8.buildROC()

    SingleNet_9 = sn.SingleNet_Train(samples,positions,20,10,8,drugResist2, sXpXa)
    SingleNet_9.trainNet(10)
    SingleNet_9.testNet(10)
    SingleNet_9.buildROC()

    SingleNet_10 = sn.SingleNet_Train(samples,positions,20,10,9,drugResist2, sXpXa)
    SingleNet_10.trainNet(10)
    SingleNet_10.testNet(10)
    SingleNet_10.buildROC()
