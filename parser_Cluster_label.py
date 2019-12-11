from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
#from itertools import zip_longest
#from jsonParser import*
import subprocess
#import SingleNet_1_Cluster_preEx as sn
import sys
import argparse
#import binaryNet_p4_preEx as bn

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
drugNeg = [0,0,0,0,0,0,0,0,0,0]
drugPos = [0,0,0,0,0,0,0,0,0,0]


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
    
for entry in drugFold:
    for idx in range(0, len(entry)):
        if(entry[idx] == 'NA'):
            drugNeg[idx] += 1
        elif(float(entry[idx]) <= 1 ):
            drugNeg[idx] += 1
        else:
            drugPos[idx] += 1
            
ind = np.arange(10)  

width = 0.35

p1 = plt.bar(ind, drugNeg, width)
p2 = plt.bar(ind, drugPos, width, bottom=drugNeg)
plt.ylabel("Samples")
plt.xticks(ind, drugList)
plt.yticks(np.arange(0,len(drugFold), len(drugFold)/5))
plt.legend((p1[0], p2[0]), ('Negative', 'Positive'), loc="upper right")
plt.savefig("/data/hibbslab/cluikart/exp50/labelDistribution.png")

