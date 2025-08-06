# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 06:10:17 2024

@author: Jesús Pueblas
"""

import numpy as np

# Get the Legendre polynomial
def Legendre(k, chi):
    p=0.
    if (k==-1):
        p = 0.
    elif (k==0):
        p = 1.
    else:
        p = (2*k-1)*1./k*chi*Legendre(k-1,chi)-(k-1)*1./k*Legendre(k-2,chi)
    return p

# Get the Legendre polynomial derivative
def Legendre_Derivative(k, chi):
    p=0.
    if (k==-1):
        p = 0
    elif (k==0):
        p = 0.
    else:
      p = (2*k-1)*1./k*(Legendre(k-1,chi)+chi*Legendre_Derivative(k-1,chi))-(k-1)*1./k*Legendre_Derivative(k-2,chi)
    return p

def getLobattoPoints(p):
    order = p+1
    coords = np.zeros(order)
    if (order==2):
      coords[0] = -1.;
      coords[1] = 1.;
    elif (order==3):
      coords[0] = -1.;
      coords[1] = 0.;
      coords[2] = 1.;
    elif (order==4):
      coords[0] = -1.;
      coords[1] = -0.4472135954999579392818;
      coords[2] =  0.4472135954999579392818;
      coords[3] = 1.;
    elif (order==5):
      coords[0] = -1.;
      coords[1] = -0.6546536707079771437983;
      coords[2] = 0.;
      coords[3] = 0.6546536707079771437983;
      coords[4] = 1.;
    elif (order==6):
      coords[0] = -1.;
      coords[1] = -0.765055323929464692851;
      coords[2] = -0.2852315164806450963142;
      coords[3] =  0.2852315164806450963142;
      coords[4] =  0.765055323929464692851;
      coords[5] =  1.;
    elif (order==7):
      coords[0] = -1.;
      coords[1] = -0.830223896278566929872;
      coords[2] = -0.4688487934707142138038;
      coords[3] =  0.;
      coords[4] =  0.4688487934707142138038;
      coords[5] =  0.830223896278566929872;
      coords[6] =  1.;
    elif (order==8):
      coords[0] = -1.;
      coords[1] = -0.8717401485096066153374;
      coords[2] = -0.5917001814331423021445;
      coords[3] = -0.2092992179024788687687;
      coords[4] =  0.2092992179024788687687;
      coords[5] =  0.5917001814331423021445;
      coords[6] =  0.8717401485096066153374;
      coords[7] =  1.;
    elif (order==9):
      coords[0] = -1.;
      coords[1] = -0.8997579954114601573123;
      coords[2] = -0.6771862795107377534459;
      coords[3] = -0.3631174638261781587108;
      coords[4] =  0.;
      coords[5] =  0.3631174638261781587108;
      coords[6] =  0.6771862795107377534459;
      coords[7] =  0.8997579954114601573123;
      coords[8] =  1.;
    elif (order==10):
      coords[0] = -1.;
      coords[1] = -0.9195339081664588138289;
      coords[2] = -0.7387738651055050750031;
      coords[3] = -0.4779249498104444956612;
      coords[4] = -0.1652789576663870246262;
      coords[5] =  0.1652789576663870246262;
      coords[6] =  0.4779249498104444956612;
      coords[7] =  0.7387738651055050750031;
      coords[8] =  0.9195339081664588138289;
      coords[9] = 1.;
    elif (order==11):
      coords[ 0] = -1.;
      coords[ 1] = -0.9340014304080591343323;
      coords[ 2] = -0.7844834736631444186224;
      coords[ 3] = -0.565235326996205006471;
      coords[ 4] = -0.2957581355869393914319;
      coords[ 5] = 0.;
      coords[ 6] = 0.2957581355869393914319;
      coords[ 7] = 0.565235326996205006471;
      coords[ 8] = 0.7844834736631444186224;
      coords[ 9] = 0.9340014304080591343323;
      coords[10] = 1.;
    elif (order==12):
      coords[ 0] = -1.;
      coords[ 1] = -0.9448992722228822234076;
      coords[ 2] = -0.8192793216440066783486;
      coords[ 3] = -0.6328761530318606776624;
      coords[ 4] = -0.3995309409653489322643;
      coords[ 5] = -0.1365529328549275548641;
      coords[ 6] = 0.1365529328549275548641;
      coords[ 7] = 0.3995309409653489322643;
      coords[ 8] = 0.6328761530318606776624;
      coords[ 9] = 0.8192793216440066783486;
      coords[10] = 0.9448992722228822234076;
      coords[11] = 1.;
    elif (order==13):
      coords[ 0] = -1.;
      coords[ 1] = -0.9533098466421639118969;
      coords[ 2] = -0.8463475646518723168659;
      coords[ 3] = -0.6861884690817574260728;
      coords[ 4] = -0.4829098210913362017469;
      coords[ 5] = -0.2492869301062399925687;
      coords[ 6] = 0.;
      coords[ 7] = 0.2492869301062399925687;
      coords[ 8] = 0.4829098210913362017469;
      coords[ 9] = 0.6861884690817574260728;
      coords[10] = 0.8463475646518723168659;
      coords[11] = 0.9533098466421639118969;
      coords[12] = 1.;
    elif (order==14):
      coords[ 0] = -1.;
      coords[ 1] = -0.9599350452672609013551;
      coords[ 2] = -0.8678010538303472510002;
      coords[ 3] = -0.7288685990913261405847;
      coords[ 4] = -0.5506394029286470553166;
      coords[ 5] = -0.3427240133427128450439;
      coords[ 6] = -0.1163318688837038676588;
      coords[ 7] = 0.1163318688837038676588;
      coords[ 8] = 0.3427240133427128450439;
      coords[ 9] = 0.5506394029286470553166;
      coords[10] = 0.7288685990913261405847;
      coords[11] = 0.8678010538303472510002;
      coords[12] = 0.9599350452672609013551;
      coords[13] = 1.;
    elif (order==15):
      coords[ 0] = -1.;
      coords[ 1] = -0.9652459265038385727959;
      coords[ 2] = -0.8850820442229762988254;
      coords[ 3] = -0.7635196899518152007041;
      coords[ 4] = -0.6062532054698457111235;
      coords[ 5] = -0.4206380547136724809219;
      coords[ 6] = -0.2153539553637942382257;
      coords[ 7] = 0.;
      coords[ 8] = 0.2153539553637942382257;
      coords[ 9] = 0.4206380547136724809219;
      coords[10] = 0.6062532054698457111235;
      coords[11] = 0.7635196899518152007041;
      coords[12] = 0.8850820442229762988254;
      coords[13] = 0.9652459265038385727959;
      coords[14] = 1.;
    elif (order==16):
      coords[ 0] = -1.;
      coords[ 1] = -0.9695680462702179329522;
      coords[ 2] = -0.8992005330934720929946;
      coords[ 3] = -0.7920082918618150639311;
      coords[ 4] = -0.6523887028824930894679;
      coords[ 5] = -0.4860594218871376117819;
      coords[ 6] = -0.2998304689007632080984;
      coords[ 7] = -0.101326273521949447843;
      coords[ 8] = 0.101326273521949447843;
      coords[ 9] = 0.2998304689007632080984;
      coords[10] = 0.4860594218871376117819;
      coords[11] = 0.6523887028824930894679;
      coords[12] = 0.7920082918618150639311;
      coords[13] = 0.8992005330934720929946;
      coords[14] = 0.9695680462702179329522;
      coords[15] = 1.;
    elif (order==17):
      coords[ 0] = -1.;
      coords[ 1] = -0.973132176631418314157;
      coords[ 2] = -0.9108799959155735956238;
      coords[ 3] = -0.8156962512217703071068;
      coords[ 4] = -0.6910289806276847053949;
      coords[ 5] = -0.5413853993301015391237;
      coords[ 6] = -0.3721744335654770419072;
      coords[ 7] = -0.1895119735183173883043;
      coords[ 8] = 0.;
      coords[ 9] = 0.1895119735183173883043;
      coords[10] = 0.3721744335654770419072;
      coords[11] = 0.5413853993301015391237;
      coords[12] = 0.6910289806276847053949;
      coords[13] = 0.8156962512217703071068;
      coords[14] = 0.9108799959155735956238;
      coords[15] = 0.973132176631418314157;
      coords[16] = 1.;
    elif (order==18):
      coords[ 0] = -1.;
      coords[ 1] = -0.9761055574121985428645;
      coords[ 2] = -0.9206491853475338738379;
      coords[ 3] = -0.8355935352180902137136;
      coords[ 4] = -0.7236793292832426813062;
      coords[ 5] = -0.5885048343186617611735;
      coords[ 6] = -0.4344150369121239753423;
      coords[ 7] = -0.2663626528782809841677;
      coords[ 8] = -0.08974909348465211102265;
      coords[ 9] = 0.08974909348465211102265;
      coords[10] = 0.2663626528782809841677;
      coords[11] = 0.4344150369121239753423;
      coords[12] = 0.5885048343186617611735;
      coords[13] = 0.7236793292832426813062;
      coords[14] = 0.8355935352180902137136;
      coords[15] = 0.9206491853475338738379;
      coords[16] = 0.9761055574121985428645;
      coords[17] = 1.;
    elif (order==19):
      coords[ 0] = -1.;
      coords[ 1] = -0.9786117662220800951526;
      coords[ 2] = -0.9289015281525862437179;
      coords[ 3] = -0.852460577796646093086;
      coords[ 4] = -0.7514942025526130141636;
      coords[ 5] = -0.6289081372652204977668;
      coords[ 6] = -0.4882292856807135027779;
      coords[ 7] = -0.3335048478244986102985;
      coords[ 8] = -0.1691860234092815713752;
      coords[ 9] = 0.;
      coords[10] = 0.1691860234092815713752;
      coords[11] = 0.3335048478244986102985;
      coords[12] = 0.4882292856807135027779;
      coords[13] = 0.6289081372652204977668;
      coords[14] = 0.7514942025526130141636;
      coords[15] = 0.852460577796646093086;
      coords[16] = 0.9289015281525862437179;
      coords[17] = 0.9786117662220800951526;
      coords[18] = 1.;
    elif (order==20):
      coords[ 0] = -1.;
      coords[ 1] = -0.9807437048939141719254;
      coords[ 2] = -0.9359344988126654357162;
      coords[ 3] = -0.8668779780899501413098;
      coords[ 4] = -0.7753682609520558704143;
      coords[ 5] = -0.6637764022903112898464;
      coords[ 6] = -0.5349928640318862616481;
      coords[ 7] = -0.3923531837139092993865;
      coords[ 8] = -0.2395517059229864951824;
      coords[ 9] = -0.0805459372388218379759;
      coords[10] = 0.0805459372388218379759;
      coords[11] = 0.2395517059229864951824;
      coords[12] = 0.3923531837139092993865;
      coords[13] = 0.5349928640318862616481;
      coords[14] = 0.6637764022903112898464;
      coords[15] = 0.7753682609520558704143;
      coords[16] = 0.8668779780899501413098;
      coords[17] = 0.9359344988126654357162;
      coords[18] = 0.9807437048939141719254;
      coords[19] = 1.;
    return coords

# Returns the value of the Lagrange polinomial from its monomial coefficient
def getLagrangeValue(monCoef,chi,i):
    Np = len(monCoef)
    Lj = 0
    for jp in range(0,Np):
        Lj += monCoef[i][jp] * chi**jp
    return Lj

# Returns the monomial coefficients for the Lagrange polynomial
def getLagrangeMonomialCoefficients(p):
    x = getLobattoPoints(p)
    Np = p+1
    s = (Np,Np)
    monCoef = np.zeros(s)
    for ip in range(0,Np):
        jfirst = 0
        if (ip==0):
            jfirst = 1
        ai = []
        ai.append(-x[jfirst])
        ai.append(1.)
        denom = x[ip] - x[jfirst]
        for jp in range(jfirst+1,Np):
            if (ip == jp):
                continue
            denom *= x[ip] - x[jp]
            aip = np.zeros(len(ai)+1)
            aip[0] = -ai[0]*x[jp]
            aip[len(aip)-1] = ai[len(ai)-1]
            for kp in range(1,len(ai)):
                aip[kp] = ai[kp-1] - x[jp] * ai[kp]
            for kp in range(0,len(ai)):
                ai[kp] = aip[kp]
            ai.append(aip[len(aip)-1])
        for jp in range(0,len(ai)):
            monCoef[ip][jp] = ai[jp]/denom
    return monCoef
    

# Returns the Lobatto's points, Lagrange polynomial derivatives and correctionn function derivatives
def getStandardElementData(p):
    order = p+1
    coords = getLobattoPoints(p)
    s = (order,order)
    Lp = np.zeros(s)
    gLp= np.zeros(order)


    for i in range(0,order):
        for j in range(0,order):
            sum = 0.
            for m in range(0,order):
                if (m!=j):
                    p = 1./(coords[j]-coords[m])
                    for l in range(0,order):
                        if (l!=m and l!=j):
                            p *= (coords[i]-coords[l])/(coords[j]-coords[l])
                    sum += p
            Lp[i][j] = sum

    for i in range(0,order):
        gLp[i] = pow(-1,order)*0.5*(Legendre_Derivative(order,coords[i])-Legendre_Derivative(order-1,coords[i]));


    return coords,Lp,gLp
