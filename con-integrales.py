# El programa que le re debo a Javi-javito


### defino las bibliotecas a usar
from re import L
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import scipy.special as sc
import matplotlib.pyplot as plt
import random

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

"""datos Tmin, Tmax, Tmean, Tmedian, Rain, Hum_mean, Hum_median"""
data=readinputdata("Oran_2007_2017.txt")
Tmin=data[:,0]
Tmax=data[:,1]
Tmean=data[:,2]
Tmedian=data[:,3]
Rain=data[:,4]
Hum_mean=data[:,5]
Hum_median=data[:,6]
#print(type(Tmin))

"""datos hogares"""
casas=readinputdata("hogares_oran.txt")
hog=casas[:,0]
hogares=sum(hog)
#print(hogares)

"""datos población"""
personas=readinputdata("censo_oran.txt")
per=personas[:,0]
poblacion=sum(per)
#poblacion=75697

""" Para los casos importados """
ref=0.
volver=0.
week=0
#importados=np.zeros(DAYS)
lectura=readinputdata("CI_ESCALON.txt")
importados=lectura[:,1]
"""
lectura=readinputdata("CI_ORAN_DET.txt")
import_case=lectura[:,1]
for i in range(DAYS):
    ref=ref+1
    if(ref>6):
        importados[i]=import_case[week]
        week=week+1
        ref=volver
"""

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

#### Calculo del EV ####
def suv_exp(t,s,rate):
    
    salida = 0.
    tau    = t - s
    
    if ( t > 0. ):
        salida = np.exp(-rate*tau)
    
    return salida

def Fbar(t,s,k,theta):
    salida = 0.
    tau = t - s

    if (t>0.):
        salida=sc.gammaincc(k,tau/theta)
    return salida


def calculo_EV(t,Tm,V_E,G_T):
    salida=0.
    tt=int(t)

    media_VE = 1. + (0.1216*Tm*Tm - 8.66*Tm + 154.79)
    var_VE = 1. + (0.1728*Tm*Tm - 12.36*Tm + 230.62)
    sigma_V = 1./media_VE
    k_VE = (media_VE*media_VE)/var_VE
    theta_VE = var_VE/media_VE
    mu_V = muerteV(Tm)*MU_MOSQUITA_ADULTA

    integral_1 = 0.
    integral_2 = 0.
    integral_3 = 0.
    integral_4 = 0.

    sigma_U_T = np.zeros(tt)
    sigma_V_1 = np.zeros(tt)
    mu_U_T = np.zeros(tt)
    U_T = np.zeros(tt)

    if (t>2.):
        for j in range(0,tt):
            T_1 = Tmean[j]
            media_VE_1 = 0.1216*T_1*T_1 - 8.66*T_1 + 154.79
            var_VE_1 = 0.1728*T_1*T_1 - 12.36*T_1 + 230.62
            sigma_V_1 = 1./media_VE_1
            k_VE_1 = (media_VE_1 * media_VE_1)/var_VE_1
            theta_VE_1 = var_VE_1/media_VE_1
            mu_V_1 = muerteV(T_1)*MU_MOSQUITA_ADULTA
            count_s=j
            sigma_U_T[j] = sigma_V_1*Fbar(t,count_s,k_VE_1,theta_VE_1)*suv_exp(t,count_s,mu_V_1)
            mu_U_T[j] = mu_V_1*Fbar(t,count_s,k_VE,theta_VE)*suv_exp(t,count_s,mu_V_1)
            U_T[j] = Fbar(t,count_s,k_VE,theta_VE)*suv_exp(t,count_s,mu_V_1)
            
        for j in range(1,tt):
            T_1 = Tmean[int(j-1)]
            media_VE_1 = 0.1216*T_1*T_1 - 8.66*T_1 + 154.79
            sigma_V_1 = 1./media_VE_1
            mu_V_1 = muerteV(T_1)*MU_MOSQUITA_ADULTA
            T_2 = Tmean[int(j)]
            media_VE_2 = 0.1216*T_2*T_2 - 8.66*T_2 + 154.79
            sigma_V_2 = 1./media_VE_2
            mu_V_2 = muerteV(T_2)*MU_MOSQUITA_ADULTA
                
            integral_1 = integral_1 + 0.5*(G_T[j-1] * U_T[j-1] + G_T[j]*U_T[j])
            integral_2 = integral_2 + 0.5*(sigma_V_1*G_T[j-1]*U_T[j-1]*sigma_V_2*G_T[j]*U_T[j])
            integral_3 = integral_3 + 0.5*(G_T[j-1]*U_T[j-1] + G_T[j]*U_T[j])
            integral_4 = integral_4 + 0.5*(mu_V_1*G_T[j-1]*U_T[j-1] + mu_V_2*G_T[j]*U_T[j])

    if (integral_1<0.):
        integral_1=0.
    if (integral_2<0.):
        integral_2=0.
    if (integral_3<0.):
        integral_3=0.
    if (integral_4<0.):
        integral_4=0.

    salida = sigma_V*V_E - sigma_V*integral_1 + integral_2 - mu_V*integral_3 + integral_4

    if (salida<0.):
        salida=0
    
    if (Tm<NO_LATENCIA):
        salida=0
    
    return salida

ht=24.
t=np.linspace(0,DAYS,DAYS)


"""sistema de ecuaciones"""
def model(z,t):
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
        muV=2*mu_V
    else:
        muV=mu_V

    ## para el epidemiologico ##
    H = Hs+He+Hi+Hr

    if(Tdeath < NO_INFECTION):
        b_theta_bv = 0.
    else:
        b_theta_bv=bite_rate*theta(Ta)*MIObv

    Sigma=1./sigma(Ta)

    G_T=np.zeros(DAYS)

    G_T[count] = bite_rate*theta(Ta)*MIObv*z[6]*z[11]/poblacion

    E_V = calculo_EV(count,Ta,Vs,G_T)

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
    dldt = mE*cg*W - mL*l - (muL + cl)*l
    dpdt = mL*l - muP*p - mP*p
    dmdt = mP*p - MU_MOSQUITO_JOVEN*m - MADURACION_MOSQUITO*m
    dvdt = 0.5*MADURACION_MOSQUITO*m - muV*v
    dVsdt = 0.5*MADURACION_MOSQUITO*m - b_theta_bv * (Hi/H)*Vs - muV*v
    dVedt = b_theta_bv * (Hi/H)*Vs - E_V - muV*Ve
    dVidt = E_V - muV*Vi
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
    return y


sol=rk4(z0,t)
Vs=sol[:,6]
HS=sol[:,9]
VI=sol[:,8]
tt=np.linspace(0,574,574)

### Calculo R0 ####
R0=np.zeros(len(t))
for i in range(DAYS):
    Tm=Tmean[i]
    VS=Vs[i]
    R0[i]=((bite_rate*theta(Tm))**2)*MIObh*MIObv*(1./Remove_expose)*(1./Remove_infect)*(1/(1+(0.1216*Tm*Tm-8.66*Tm+154.79)))*VS/ALPHA
    if (R0[i]<0): R0[i]=0.
    

ref=0
volver=0
### Casos por dia ###
casos_semana=np.zeros(574)
casos_dia=np.zeros(len(t))
for i in range(DAYS):
    Tm=Tmean[i]
    Hsus=HS[i]
    Vin=VI[i]
    c=bite_rate*theta(Tm)*MIObh*(Hsus/poblacion)*Vin
    casos_dia[i]=np.round(c,decimals=2)


reor=casos_dia.reshape((574,7))
#print(vec)
for i in range(0,574):
    for j in range(0,7):
        ref = ref + reor[i][j]
    casos_semana[i]=ref
    ref=volver

v=sol[:,5]

#for i in range(0,DAYS):
#    print(v[i])
#print(np.size(v))
vec=R0.reshape((574,7))
#print(vec)
R0_week=np.zeros(574)
for i in range(0,574):
    for j in range(0,7):
        ref = ref + vec[i][j]
    R0_week[i]=ref
    ref=volver


np.savetxt("CS_integrales.txt",casos_semana,fmt='%1.4e') #casos por semana
np.savetxt("CD_integrales.txt",casos_dia,fmt='%1.4e') #casos por dia

np.savetxt("hum_sus_integrales.txt", HS,fmt='%1.4e') #humanos susceptibles
np.savetxt("vec_in_integrales.txt", VI,fmt='%1.4e')  #vectores infectados


#plt.plot(t,D,'b',t,W,'r',t,l,'g',t,p,'c',t,m,'m',t,v,'y')
"""
plt.plot(tt,R0_week/7.)
plt.ylabel('R0')
plt.xlabel('dia')
plt.show()
"""

fig,ax=plt.subplots(2,1)

ax[0].plot(tt,R0_week/7,color='r',label='R0')

ax[1].plot(t,v/poblacion,color='m',label='V/H')

ax[0].legend()
ax[1].legend()
plt.show()