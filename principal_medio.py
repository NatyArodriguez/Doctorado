

### defino las bibliotecas a usar
from re import L
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
from semanas_epi import orden

# Primero voy a intentar resolver el modelo ecológico para la población del aedes

#defino las mismas variables que el programa original
"""tasa de gente importada infectada"""
EGG_LIFE=120.
EGG_LIFE_wet=90.

RATE_IMPORT=0.12
beta_day=2.82
bite_rate=0.25

"""prevalencia cero y population"""
ALPHA=0.75
MIObh=0.75
MIObv=0.75
Remove_infect=7.
Remove_expose=5.
RATE_CASOS_IMP=2.
MATAR_VECTORES=12.5
NO_INFECCION=15.
NO_LATENCIA=15.
MUERTE_ACUATICA=0.5
TEMP_ACUATICA=10.
"""efectividad del control biologico"""
EFECTIV=0.3
EFIC_LP=0.3
MATAR_VECTORES=12.5
EFECT_V=0.5
NO_INFECTION=22.
NO_LATENCIA=16.

"""parametros aedes en dias"""
MU_MOSQUITO_JOVEN=1./2.
MADURACION_MOSQUITO=1./2.
MU_MOSQUITA_ADULTA=1./10.
Ve_Lat=4
Kmax=190.
DIFUSION_VECTORES=0.0
SUV=0.6
COND_I=10
LAG=3
AV_EF_TEMP=0.5
AV_EF_HUM=0.5
Rthres=12.5
Hmax=24.
kmmC=3.3E-6

""""parametros tartagal?"""
POPULATION=3.1E6
RAD_CENSAL=60
MAX_VECI=11
#DAYS=4018
WEEKS=575
BEGIN_YEAR=2007
YEARS=11
Dt=1.

"""limite de control biologico"""
LIMPIA=0
LARVICIDA=0

"""indices de control biologico"""
INDICE_Larv=1
INDICE_Pup=1


mu_D=1./EGG_LIFE
mu_W=1./EGG_LIFE_wet


############# Extraigo los valores de los archivos #############
def readinputdata(filename):
    fichero=open(filename,'r')
    f=[]
    line='0'
    while len(line)>0:
        line=np.array(fichero.readline().split()).astype(float)
        if len(line)>0:
            f.append(line)
    fichero.close()
    return np.array(f)



########################### LECTURA DE DATOS ##################################
"""datos Tmin, Tmax, Tmean, Tmedian, Rain, Hum_mean, Hum_median"""
data=readinputdata("Oran_2001_2017.txt")
Tmin=data[:,0]
Tmax=data[:,1]
Tmean=data[:,2]
Tmedian=data[:,3]
Rain=data[:,4]
Hum_mean=data[:,5]
Hum_median=data[:,6]

"""datos hogares"""
casas=readinputdata("hogares_oran.txt")
hog=casas[:,0]
hogares=sum(hog)
#hogares=17633

"""datos población"""
personas=readinputdata("censo_oran.txt")
per=personas[:,0]
poblacion=sum(per)
#poblacion=75697


"""datos clinicos"""
lala=readinputdata("datos_clinicos_2010_2016.txt")
casos_clinicos=lala[:,0]


""" Para los casos importados """
ref=0.
volver=0.
week=0
lectura=readinputdata("CI_2001_2017.txt")
importados=lectura[:,0]
DAYS=len(importados)

########################################################################################

""" Definición de funciones """
def CG(larv,k):
    if (larv<k):
        a=1-(larv/k)
    else: a=0
    return (a)

def egg_wet(lluvia):
    return 0.8*(lluvia/Rthres)**5/(1. + (lluvia/Rthres)**5)

def theta(Tm):
    if ( (11.7<Tm) and (Tm<32.7) ):
        b=0.1137*(-5.4 + 1.8*Tm - 0.2124 *Tm*Tm + 0.01015*Tm*Tm*Tm-0.0001515*Tm*Tm*Tm*Tm)
    else:
        b=0.
    return(b)

def sigma(Tm):
    return 0.1216*Tm*Tm - 8.66*Tm + 154.79

def ht1(lluvia,humedo,ht,Tm):
    aux= ht + lluvia - kmmC*(25. + Tm)*(25. + Tm)*(100. - humedo)
    if (aux<0.):
        salida=0.
    else:
        if(Hmax<=aux):
            salida=Hmax
        else:
            salida=aux
    return(salida)

def rate_mx(Tm,DHA,DHH,T12):
    aux0=(DHA/1.987207)*(1/298. - 1./(Tm+273.15))
    aux1=(DHH/1.987207)*(1./T12 - 1./(Tm+273.15))
    salida=((Tm+273.15)/298.)*(np.exp(aux0)/(1+np.exp(aux1)))
    return(salida)

def muerteV(Tm):
    factor=0.0360827
    salida=8.692E-1-(1.59E-1)*Tm+(1.116E-2)*Tm*Tm-(3.408E-4)*Tm*Tm*Tm+(3.809E-6)*Tm*Tm*Tm*Tm
    return salida/factor

ht=24.
t=np.linspace(0,DAYS,DAYS)


"""sistema de ecuaciones"""
def model(z,t):
    count=int(t)
    D = z[0] #HUEVOS SECOS
    W = z[1] #HUEVOS HUMEDOS
    l = z[2] #LARVAS
    p = z[3] #PUPAS
    m = z[4] #MOSQUITOS
    v = z[5] #VECTOR
    Vs = z[6] #VECTOR SUSCEPTIBLE
    Ve = z[7] #VECTOR EXPUESTO
    Vi = z[8] #VECTOR INFECTADO
    Hs = z[9] #HOSPEDADOR SUSCEPTIBLE
    He = z[10] #HOSPEDADOR EXPUESTO
    Hi = z[11] #HOSPEDADOR INFECTADO
    Hr = z[12] #HOSPEDADOR RECUPERADO

    ht=24.


    Ta=Tmean[count]
    T_min=Tmin[count]

    r=Rain[count]

    hum=Hum_mean[count]

    Tdeath=Tmin[count]

    thetaT0=beta_day*theta(Ta)

    fR=egg_wet(r)

    Larv= l

    ht=ht1(r,hum,ht,Ta)

    KL=hogares*(170*ht/Hmax + 1.)

    mE=0.24*rate_mx(Ta,10798.,100000.,14184.)

    cg=CG(Larv,KL)

    if(Tdeath<13.4):
        mL=0.
    else:
        mL=0.2088*rate_mx(Ta,26018.,55990.,304.6)
            
    muL=0.01 + 0.9725*np.exp(-(Ta - 4.85)/2.7035)

    cl=1.5*(Larv/KL)

    mP=0.384*rate_mx(Ta,14931.,-472379.,148.)

    muP=0.01 + 0.9725*np.exp(-(Ta-4.85)/2.7035)

    mu_V=muerteV(Ta)*MU_MOSQUITA_ADULTA
        
    if(Tdeath<MATAR_VECTORES):
        muV=2.0*mu_V
    else:
        muV=mu_V

    ## para el epidemiologico ##
    H = Hs+He+Hi+Hr

    if(Tdeath < NO_INFECTION):
        b_theta_bv = 0.
    else:
        b_theta_bv=bite_rate*theta(Ta)*MIObv
        
    Sigma=1./sigma(Ta)

    EV=Sigma*Ve
    if(EV < 0.): EV=0.
    if(Ta < NO_LATENCIA): EV=0.

    if(Tdeath < NO_INFECTION):
        b_theta_bh=0.
    else:
        b_theta_bh=bite_rate*theta(Ta)*MIObh


    sigmaH=1./Remove_expose

    gama= 1./Remove_infect

    delta_I= RATE_CASOS_IMP * importados[count]


    dDdt = thetaT0*v - mu_D*D - fR*D
    dWdt = fR*D - mu_W*W - mE*cg*W

    if (T_min<TEMP_ACUATICA):
        dWdt=MUERTE_ACUATICA*dWdt

    dldt = mE*cg*W - mL*l - (muL + cl)*l

    if (T_min<TEMP_ACUATICA):
        dldt=MUERTE_ACUATICA*dldt

    dpdt = mL*l - muP*p - mP*p

    if (T_min<TEMP_ACUATICA):
        dpdt=MUERTE_ACUATICA*dpdt

    dmdt = mP*p - MU_MOSQUITO_JOVEN*m - MADURACION_MOSQUITO*m
    dvdt = 0.5*MADURACION_MOSQUITO*m - muV*( Vs + Ve + Vi )
    dVsdt = 0.5*MADURACION_MOSQUITO*m - b_theta_bv * (Hi/H)*Vs - muV*Vs
    dVedt = b_theta_bv * (Hi/H)*Vs - EV - muV*Ve
    dVidt = EV - muV*Vi
    dHsdt = -b_theta_bh * (Hs/H)*Vi
    dHedt = b_theta_bh * (Hs/H)*Vi - sigmaH*He
    dHidt = sigmaH*He - gama*Hi + delta_I
    dHrdt = gama*Hi
    return np.array([dDdt,dWdt,dldt,dpdt,dmdt,dvdt,dVsdt,dVedt,dVidt,dHsdt,dHedt,dHidt,dHrdt])

"""Condiciones iniciales de los bichos"""
bichos=readinputdata("bichos.txt")
secos=bichos[:,0]
mojados=bichos[:,1]
larva=bichos[:,2]
pupa=bichos[:,3]
maduro=bichos[:,4]
H_t_0=24.#bichos[:,5]

ED0=sum(secos)
EW0=sum(mojados)
L0=sum(larva)
P0=sum(pupa)
M0=sum(maduro)

z0 = np.zeros(13)

z0[0] = ED0
z0[1] = EW0
z0[2] = L0
z0[3] = P0
z0[4] = M0
z0[5] = 0.
z0[6] = 0.
z0[7] = 0.
z0[8] = 0.
z0[9] = ALPHA*poblacion
z0[10] = 0.
z0[11] = 0.
z0[12] = poblacion - z0[9] - z0[10] - z0[11]


def euler(y0,t):
    n=len(t)
    y=np.zeros((n,len(y0)))
    y[0]=y0
    for i in range(n-1):
        y[i+1]=y[i]+(t[i+1]-t[i])*model(y[i],t[i])
        if(y[i][0]<0):y[i][0]=0
        if(y[i][1]<0):y[i][1]=0
        if(y[i][2]<0):y[i][2]=0
        if(y[i][3]<0):y[i][3]=0
        if(y[i][4]<0):y[i][4]=0
        if(y[i][5]<0):y[i][5]=0
        if(y[i][6]<0):y[i][6]=0
        if(y[i][7]<0):y[i][7]=0
        if(y[i][8]<0):y[i][8]=0
        if(y[i][9]<0):y[i][9]=0
        if(y[i][10]<0):y[i][10]=0
        if(y[i][11]<0):y[i][11]=0
        if(y[i][12]<0):y[i][12]=0
    return y

def rk4(y0,t):
    n=len(t)
    y=np.zeros((n,len(y0)))
    y[0]=y0
    for i in range(n-1):
        h=Dt
        k1=model(y[i],t[i])
        k2=model(y[i] + k1*h/2., t[i] + h/2.)
        k3=model(y[i] + k2*h/2., t[i] + h/2.)
        k4=model(y[i] + k3*h,t[i] + h)
        y[i+1]=y[i] + (h/6.)* (k1+2*k2+2*k3+k4)
        if(y[i][0]<0):y[i][0]=0
        if(y[i][1]<0):y[i][1]=0
        if(y[i][2]<0):y[i][2]=0
        if(y[i][3]<0):y[i][3]=0
        if(y[i][4]<0):y[i][4]=0
        if(y[i][5]<0):y[i][5]=0
        if(y[i][6]<0):y[i][6]=0
        if(y[i][7]<0):y[i][7]=0
        if(y[i][8]<1):y[i][8]=0
        if(y[i][9]<0):y[i][9]=0
        if(y[i][10]<0):y[i][10]=0
        if(y[i][11]<0):y[i][11]=0
        if(y[i][12]<0):y[i][12]=0
    return y


############## RESULTADOS DEL SISTEMA ###########
sol=rk4(z0,t)
Vs=sol[:,6]
HS=sol[:,9]
VI=sol[:,8]
H_SECOS=sol[:,0]
H_HUM=sol[:,1]
#################################################

### Calculo R0 ####
R0=np.zeros(len(t))
for i in range(DAYS):
    Tm=Tmean[i]
    VS=Vs[i]
    Hs=HS[i]
    MUv=muerteV(Tm)*MU_MOSQUITA_ADULTA
    R0[i]=((bite_rate*theta(Tm))**2)*MIObh*MIObv*(1./MUv)*(1.-np.exp(-(1/sigma(Tm))/MUv))*(Hs/poblacion)*Remove_infect*(VS/poblacion)
    if (R0[i]<0.): R0[i]=0.


ref=0
volver=0
### Casos por dia ###
casos_dia=np.zeros(len(t))
for i in range(DAYS):
    Tm=Tmean[i]
    Tminimo=Tmin[i]
    Hsus=HS[i]
    Vin=VI[i]
    c=bite_rate*theta(Tm)*MIObh*(Hsus/poblacion)*Vin
    if (Tminimo<12):
        casos_dia[i]=0.
    else:
        casos_dia[i]=np.round(c,decimals=2)


################# GENERAR ARCHIVOS ###############################
np.savetxt("R0_dia_2001.txt",R0,fmt='%1.4e')
np.savetxt("CD_2001.txt",casos_dia,fmt='%1.4e') #casos por dia

np.savetxt("hum_sus_2001.txt", HS,fmt='%1.4e') #humanos susceptibles
np.savetxt("vec_in_2001.txt", VI,fmt='%1.4e')  #vectores infectados
np.savetxt("vec_sus_2001.txt", Vs,fmt='%1.4e') #vectores susceptibles
np.savetxt("huevos_secos_2001.txt", H_SECOS,fmt='%1.4e') #huevos secos
np.savetxt("huevos_humedos_2001.txt",H_HUM,fmt='%1.4e') #huevos humedos
#################################################################