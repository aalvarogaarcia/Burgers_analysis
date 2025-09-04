# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:24:55 2024

@author: Jesús Pueblas
"""

from sys import argv
import matplotlib.pyplot as plt
import numpy as np

def ReadBlockData(document,istart,beginLabel,endLabel):
    data = []
    N=len(document)
    istart0=istart
    for i in range(istart,N):
        line = document[i]
        fields = line.split()
        nfields = len(fields)
        if (nfields>0):
            if (fields[0] == beginLabel):
                istart=i+1
                break
    if (istart==istart0):
        data = []
        print("First label not found: ",beginLabel)
        return istart,data

    foundEndLabel="FALSE"
    iend=istart
    for i in range(istart,N):
        line = document[i]
        fields = line.split()
        nfields = len(fields)
        if (nfields>0):
            if (fields[0] != endLabel):
                data.append(line)
            else:
                iend = istart
                foundEndLabel="TRUE"
                break
    if (foundEndLabel=="FALSE"):
        data = []
        print("Second label not found: ",endLabel)
        return istart,data
    return iend,data


inputfile=open(argv[1],'r')
document = inputfile.readlines()
inputfile.close()

finish = "FALSE"

istart=0
while (finish=="FALSE"):
    iend,data = ReadBlockData(document, istart, "BEGIN_BLOCK", "END_BLOCK")
    istart=iend
    if (len(data)==0):
        finish="TRUE"
    else:
        C = int(data[0].split()[0])
        p = int(data[1].split()[0])
        LABEL = data[2].split()[0]
        dof = np.zeros(len(data)-3)
        errors = np.zeros(len(data)-3)
        for i in range(3,len(data)):
            line = data[i]
            fields = line.split()
            dofi = 1./float(fields[0])
            errori = float(fields[1])
            dof[i-3] = dofi
            errors[i-3] = errori
        plt.plot(dof,errors,'o-',label=LABEL)
        doftrend = np.zeros(2)
        l2etrend = np.zeros(2)
        doftrend[0] = dof[0]
        doftrend[1] = dof[-1]
        l2etrend[0] = C*doftrend[0]**(p+1)
        l2etrend[1] = C*doftrend[1]**(p+1)
        plt.plot(doftrend,l2etrend,'--')


plt.xlabel('1/dof')
plt.ylabel('l2norm(error)')
plt.legend(loc='upper left')
plt.xscale("log")
plt.yscale("log")
plt.show()
