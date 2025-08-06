# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 06:31:34 2024

@author: Jesús Pueblas
"""

#!/usr/bin/python

import numpy as np

# Get the value for a label in a document of the form:
# Inputs:
# document : list with lines of the input file previously read
# LABEL    : label to extract its value
def getValueFromLabel(document,LABEL):
    for line in document:
        fields = line.split()
        nfields = len(fields)
        if (nfields>0):
            if (fields[0] == LABEL):
                if (nfields > 1):
                    return fields[1]
                else:
                    print("Error reading value of label: ",LABEL)
    print("Error! Label: ",LABEL," not found!")
    return "ERROR"

# Add a block of data to a document
def AddBlockData(document,beginLabel,endLabel,data):
    document.append(beginLabel+"\n")
    for line in data:
        document.append(line+"\n")
    document.append(endLabel+"\n")

# Read a block of data from a document
def ReadBlockData(document,beginLabel,endLabel):
    data = []
    istart=-1
    N=len(document)
    for i in range(0,N):
        line = document[i]
        fields = line.split()
        nfields = len(fields)
        if (nfields>0):
            if (fields[0] == beginLabel):
                istart=i+1
                break
    if (istart==-1):
        print("First label not found: ",beginLabel)

    foundEndLabel="FALSE"
    for i in range(istart,N):
        line = document[i]
        fields = line.split()
        nfields = len(fields)
        if (nfields>0):
            if (fields[0] != endLabel):
                data.append(line)
            else:
                foundEndLabel="TRUE"
                break
    if (foundEndLabel=="FALSE"):
        print("Second label not found: ",endLabel)
    return data

# Get the solution from the document
def GetMeshAndSolution(document):
    Uini = ReadBlockData(document,"BEGIN_SOLUTION","END_SOLUTION")
    nnode = len(Uini)
    u = np.zeros(nnode)
    x = np.zeros(nnode)
    for i in range(0,nnode):
        line = Uini[i]
        fields = line.split()
        if (len(fields) != 2):
            print("Error reading initial solution, fields different from 6")
            exit()
        x[i] = float(fields[0])
        u[i] = float(fields[1])
    return x,u

# Add a single data of integer type
def AddSingleDataInt(document,label,value):
    line = label  + " " + '%i' % value + "\n"
    document.append(line)
    
# Add a single data of float type
def AddSingleDataFloat(document,label,value):
    line = label + " " + '%f' % value + "\n"
    document.append(line)

# Add a single data with exponential format
def AddSingleDataExp(document,label,value):
    line = label  + " " + '%e' % value + "\n"
    document.append(line)


# Write the input data of the simulation:
def WriteInputData(document,N,p,v,Nref,IniS,dt,tsim,Ndump, use_les=False, sgs_params=None, Nx=None, Ny=None):
    newLine = "# Input file for fr-burgers-turbulent.py program\n"
    document.append(newLine)

    newLine="N      " + '%15i' % N     + " # Number of mesh points\n"
    document.append(newLine)
    
    if Nx is not None:
        newLine = "NX     " + '%15i' % Nx + " # Number of mesh points in X\n"
        document.append(newLine)
    if Ny is not None:
        newLine = "NY     " + '%15i' % Ny + " # Number of mesh points in Y\n"
        document.append(newLine)
    
    newLine="NREF   " + '%15i' % Nref  + " # Number of refinement levels of the mesh\n"
    document.append(newLine)
    newLine="P      " + '%15i' % p     + " # Polynomial degree for FR scheme, which is the degree of the lagrange polynomial\n"
    document.append(newLine)
    newLine="VISC   " + '%15f' % v     + " # Viscosity\n"
    document.append(newLine)
    newLine="INISOL " + '%15s' % IniS  + " # Initial solution type: SINE,GAUSSIAN,SQUARE,TURBULENT\n"
    document.append(newLine)
    newLine="DT     " + '%15f' % dt    + " # Time step\n"
    document.append(newLine)
    newLine="TSIM   " + '%15f' % tsim  + " # Maximum number of iterations\n"
    document.append(newLine)
    newLine="NDUMP  " + '%15i' % Ndump + " # Interval of iterations to dump a solution\n"
    document.append(newLine)
    
    # Escribir parámetros LES si se usan
    document.append("# --- LES Parameters ---\n")
    newLine = "USE_LES         " + '%15s' % str(use_les).upper() + "\n"
    document.append(newLine)
    if use_les and sgs_params is not None:
        newLine = "SGS_MODEL_TYPE  " + '%15s' % sgs_params.get('model_type', 'N/A') + "\n"
        document.append(newLine)
        newLine = "SGS_FILTER_RATIO " + '%15.2f' % sgs_params.get('filter_width_ratio', 0.0) + "\n"
        document.append(newLine)
        newLine = "SGS_AVG_TYPE    " + '%15s' % sgs_params.get('avg_type', 'N/A') + "\n"
        document.append(newLine)
        newLine = "SGS_CS_MIN      " + '%15.4f' % sgs_params.get('Cs_min', 0.0) + "\n"
        document.append(newLine)
    document.append("\n")

# Write the solution file
def WriteFile_1D(filename,x,U,N,p,v,Nref,IniS,dt,tsim,Ndump, use_les=False, sgs_params=None):
    print("Dumping solution ....")
    nnode = len(x)
    # Few postprocess of the solution
    usol = []

    for i in range(0,nnode):
        line = '%0.15e' % x[i] + " " + '%0.15e' % U[i]
        usol.append(line)

    # Write the results
    document = []
    WriteInputData(document,N,p,v,Nref,IniS,dt,tsim,Ndump, use_les, sgs_params)
    AddBlockData(document,"BEGIN_SOLUTION","END_SOLUTION",usol)

    #Finally, write the document
    outputfile = open(filename,'w')
    outputfile.writelines(document)
    outputfile.close()
    
    
def WriteFile_2D(filename, x, y, U, Nx, Ny, p, v, Nref, IniS, dt, tsim, Ndump):
    """
    Escribe el archivo de solución para una simulación 2D.
    Guarda 4 columnas: x, y, u, v.
    """
    print(f"Dumping 2D solution to {filename}....")
    num_nodes = len(x)
    u_view = U[0:num_nodes]
    v_view = U[num_nodes:]
    
    solution_lines = []
    for i in range(num_nodes):
        line = f"{x[i]:.15e} {y[i]:.15e} {u_view[i]:.15e} {v_view[i]:.15e}\n"
        solution_lines.append(line)

    document = []
    # Usamos la versión actualizada de WriteInputData
    WriteInputData(document, -1, p, v, Nref, IniS, dt, tsim, Ndump, Nx=Nx, Ny=Ny)
    AddBlockData(document, "BEGIN_SOLUTION", "END_SOLUTION", solution_lines)

    with open(filename, 'w') as outputfile:
        outputfile.writelines(document)

