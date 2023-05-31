from ast import Del
from re import L
import numpy as np
import pandas as pd
import optuna
from scipy.optimize import leastsq
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
from semanas_epi2 import orden
from selector_datos import seleccion
from sklearn.metrics import mean_squared_error



"""tasa de gente importada infectada"""
EGG_LIFE=120.
EGG_LIFE_wet=90.

#RATE_IMPORT=0.12
#beta_day=2.82
#bite_rate=0.2114

"""prevalencia cero y population"""
ALPHA=0.75
MIObh=0.75
MIObv=0.75
Remove_infect=7.
Remove_expose=5.
RATE_CASOS_IMP=1.
MATAR_VECTORES=12.5
NO_INFECCION=15.
NO_LATENCIA= 16.
MUERTE_ACUATICA=0.5
TEMP_ACUATICA=10.

"""efectividad del control biologico"""
EFECTIV=0.3
EFIC_LP=0.3
MATAR_VECTORES=12.5
EFECT_V=0.5
NO_INFECTION=15.
NO_LATENCIA=16.

"""parametros aedes en dias"""
MU_MOSQUITO_JOVEN=1./2.
MADURACION_MOSQUITO=1./2.
MU_MOSQUITA_ADULTA=1./10.
Ve_Lat=4
Kmax=120.
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
DAYS=4018
WEEKS=574
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

"""datos Tmin, Tmax, Tmean, Tmedian, Rain, Hum_mean, Hum_median"""
data=readinputdata("Oran_2007_2017.txt")
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

"""datos población"""
personas=readinputdata("censo_oran.txt")
per=personas[:,0]
poblacion=sum(per)
#poblacion=75697

"""datos clinicos"""
data=readinputdata("datos_clinicos.txt")
casos_clinicos=data[:,0]

""" Para los casos importados """
ref=0.
volver=0.
week=0
#importados=np.zeros(DAYS)
lectura=readinputdata("series_importados.txt")
importados=lectura[:,7]
inicio = [1,'ene',2007]
final = [31,'dic',2017]
importados= seleccion(importados,inicio,final)


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
def model(z,t,b_day,b_pica):
    count=int(t)
    D = z[0]
    W = z[1]
    l = z[2]
    p = z[3]
    m = z[4]
    v = z[5]
    Vs = z[6]
    Ve = z[7]
    Vi = z[8]
    Hs = z[9]
    He = z[10]
    Hi = z[11]
    Hr = z[12]

    ht=24.

    Ta=Tmean[count]
    T_min=Tmin[count]

    r=Rain[count]

    hum=Hum_mean[count]

    Tdeath=Tmin[count]

    thetaT0=b_day*theta(Ta)

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
        muV=2*mu_V
    else:
        muV=mu_V

    ## para el epidemiologico ##
    H = Hs+He+Hi+Hr

    if(Tdeath < NO_INFECTION):
        b_theta_bv = 0.
    else:
        b_theta_bv=b_pica*theta(Ta)*MIObv

    Sigma=1./sigma(Ta)

    EV=Sigma*Ve
    if(EV < 0.): EV=0.
    if(Ta < NO_LATENCIA): EV=0.

    if(Tdeath < NO_INFECTION):
        b_theta_bh=0.
    else:
        b_theta_bh=b_pica*theta(Ta)*MIObh

    sigmaH=1./Remove_expose

    gama= 1./Remove_infect

    delta_I= RATE_CASOS_IMP*importados[count]


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
    dvdt = 0.5*MADURACION_MOSQUITO*m - muV*v
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

D=np.empty_like(t)
W=np.empty_like(t)
l=np.empty_like(t)
p=np.empty_like(t)
m=np.empty_like(t)
v=np.empty_like(t)

D[0]=z0[0]
W[0]=z0[1]
l[0]=z0[2]
p[0]=z0[3]
m[0]=z0[4]
v[0]=z0[5]

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
        if(y[y][7]<0):y[i][7]=0
        if(y[i][8]<0):y[i][8]=0
        if(y[i][9]<0):y[i][9]=0
        if(y[i][10]<0):y[i][10]=0
        if(y[i][11]<0):y[i][11]=0
        if(y[i][12]<0):y[i][12]=0
    return y

def rk4(f,y0,t,args=()):
    n=len(t)
    y=np.zeros((n,len(y0)))
    y[0]=y0
    for i in range(n-1):
        h=Dt
        k1=f(y[i],t[i],*args)
        k2=f(y[i] + k1*h/2., t[i] + h/2.,*args)
        k3=f(y[i] + k2*h/2., t[i] + h/2.,*args)
        k4=f(y[i] + k3*h,t[i] + h,*args)
        y[i+1]=y[i] + (h/6.)* (k1+2*k2+2*k3+k4)
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


### aqui defino la función a utilizar para los cuadrados mínimos ###

def fun(parametros):
    pone=parametros[0] #beta_day
    pica=parametros[1] #bite rate
    t=np.linspace(0,DAYS,DAYS)
    c_sim_week=np.zeros(WEEKS)
    ref=0.
    volver=0.
    contar=0.
    
    solucion=rk4(model,z0,t,args=(pone,pica))

    vec_in=solucion[:,8]
    hosp_sus=solucion[:,9]
    #np.savetxt("hum_sus.txt", hosp_sus,fmt='%1.4e') #humanos susceptibles
    #np.savetxt("vec_in.txt", vec_in,fmt='%1.4e')  #vectores infectados
    c_sim=np.zeros(DAYS)

    for i in range(0,DAYS):
        Tm=Tmean[i]
        c_sim[i]=pica*MIObh*theta(Tm)*vec_in[i]*hosp_sus[i]/poblacion

    reor=c_sim.reshape((WEEKS,7))
    for i in range(0,WEEKS):
        for j in range(0,7):
            ref = ref + reor[i][j]
            c_sim_week[i]=np.round(ref,decimals=2)
            ref=volver
    
    c_sim_mes=orden(c_sim_week)
    return c_sim_mes 

def diferencia(parametros,y):
    return y - fun(parametros)

def objetive(trial):
    x = trial.suggest_float("x",1,2.8) #beta day
    y = trial.suggest_float("y",0,1) #bite rate
    parametros=[x,y]
    y_pred = fun(parametros)
    y_val = casos_clinicos
    error = mean_squared_error(y_val,y_pred)
    return error

#search_space = {"x":[1,2.8], "y":[0,0.5]}
#study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))


study = optuna.create_study(direction = 'minimize')
study.optimize(objetive, n_trials = 100)

best_params = study.best_params

found_x = best_params["x"]
found_y = best_params["y"]

print("FOUND beta_day:{}, b_r: {}".format(found_x,found_y))
