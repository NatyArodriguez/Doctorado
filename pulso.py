import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randint
import numpy as np
from matplotlib import animation
from selector_datos import seleccion


###################### Animacion para el pulso ####################
'''
fig = plt.figure()
ax = plt.axes(xlim=(0, 11), ylim=(0, 6))
line, = ax.plot([], [], lw=3)
line1, = ax.plot([],[], lw=2)
# initialization function
def init():
    line.set_data([], [])
    return line,

def señal(x,tau,rango): # x=altura, tau=periodo
    val=[]
    inicio=0
    volver=0
    for i in range(0,rango):
        inicio=inicio + 1
        if (inicio==tau):
            val.append(x)
        else:
            val.append(0)
    return (np.array(val))


def animacion(i):
    x = np.linspace(0,20,20)
    y = señal(3,i,np.size(x))
    z = señal(5,i+3,np.size(x))
    line.set_data(x,y)
    #line1.set_data(x,z)
    return line, line1 

anim = animation.FuncAnimation(fig, animacion, init_func=init,
                               frames=15, interval=100, blit=True)
ax.set_title('Pulso')
plt.show()
anim.save('pulso.gif', writer = animation.PillowWriter(fps=60))
'''
################################################################

###################################################################
##### Funciones ####

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


def señal(x,tau,rango): # x=altura, tau=periodo
    val=[]
    inicio=0
    volver=0
    for i in range(0,rango):
        inicio=inicio + 1
        if (inicio==tau):
            val.append(x)
        else:
            val.append(0)
    return (np.array(val))

##################################################
EGG_LIFE=120.
EGG_LIFE_wet=90.

RATE_IMPORT=0.12
beta_day=2.82
bite_rate=0.25

ALPHA=0.75
MIObh=0.75
MIObv=0.75
Remove_infect=7.
Remove_expose=5.
RATE_CASOS_IMP=2.
MATAR_VECTORES=12.5
NO_INFECTION=22.
NO_LATENCIA=15.
MUERTE_ACUATICA=0.5
TEMP_ACUATICA=10.
MU_MOSQUITO_JOVEN=1./2.
MADURACION_MOSQUITO=1./2.
MU_MOSQUITA_ADULTA=1./10.
Kmax=190.
Rthres=12.5
kmmC=3.3E-6
mu_D=1./EGG_LIFE
mu_W=1./EGG_LIFE_wet
Dt=1.
Hmax=24.


##### establecer temporada ######

inicio = [1,'dic',2016]
final = [31,'may',2017]
tiempo1=np.arange('2016-12-01','2017-06-01',dtype='datetime64[D]')
a = inicio[2]
b= final[2]

data=readinputdata("Oran_2001_nuevo.txt")
tmin=data[:,0]
tmax=data[:,1]
tmean=data[:,2]
tmedian=data[:,3]
rain=data[:,4]
hum_mean=data[:,5]
hum_median=data[:,6]

Tmin=seleccion(tmin,inicio,final)
Tmax=seleccion(tmax,inicio,final)
Tmean=seleccion(tmean,inicio,final)
Tmedian=seleccion(tmedian,inicio,final)
Rain=seleccion(rain,inicio,final)
Hum_mean=seleccion(hum_mean,inicio,final)
Hum_median=seleccion(hum_median,inicio,final)

DAYS=np.size(Tmin)
tiempo=np.linspace(0,DAYS,DAYS)

casas=readinputdata("hogares_oran.txt")
hog=casas[:,0]
hogares=sum(hog)
#hogares=17633

personas=readinputdata("censo_oran.txt")
per=personas[:,0]
poblacion=sum(per)
#poblacion=75697


DAYS=np.size(Tmin)
lectura=readinputdata("CI_ESCALON.txt")
importados=lectura[:,1]
aaa = seleccion(importados,[1,'ene',2001],[31,'dic',2017])
casos_importados=aaa
#casos_importados=señal(10,57,DAYS)

########################################################################################


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
    return (0.1216*Tm*Tm - 8.66*Tm + 154.79)

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
        
    Sigma=1./(1. + sigma(Ta))

    EV=Sigma*Ve
    if(EV < 0.): EV=0.
    if(Ta < NO_LATENCIA): EV=0.

    if(Tdeath < NO_INFECTION):
        b_theta_bh=0.
    else:
        b_theta_bh=bite_rate*theta(Ta)*MIObh


    sigmaH=1./Remove_expose

    gama= 1./Remove_infect

    delta_I= RATE_CASOS_IMP * casos_importados[count]


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
    dVsdt = 0.5*MADURACION_MOSQUITO*m - b_theta_bv * (Hi/poblacion)*Vs - muV*Vs
    dVedt = b_theta_bv * (Hi/poblacion)*Vs - EV - muV*Ve
    dVidt = EV - muV*Vi
    dHsdt = -b_theta_bh * (Hs/poblacion)*Vi
    dHedt = b_theta_bh * (Hs/poblacion)*Vi - sigmaH*He
    dHidt = sigmaH*He - gama*Hi + delta_I
    dHrdt = gama*Hi
    return np.array([dDdt,dWdt,dldt,dpdt,dmdt,dvdt,dVsdt,dVedt,dVidt,dHsdt,dHedt,dHidt,dHrdt])


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

z = np.zeros(13)

z[0] = ED0
z[1] = EW0
z[2] = L0
z[3] = P0
z[4] = M0
z[5] = 0.
z[6] = 0.
z[7] = 0.
z[8] = 0.
z[9] = ALPHA*poblacion
z[10] = 0.
z[11] = 0.
z[12] = poblacion - z[9] - z[10] - z[11]

def RK4(t,h,x):
    n = len(x)
    salida = np.zeros(n)
    
    k1 = np.zeros(n)
    k2 = np.zeros(n)
    k3 = np.zeros(n)
    k4 = np.zeros(n)

    k1 = h*model(x,t)
    k2 = h*model(x + 0.5*k1, t + 0.5*h)
    k3 = h*model(x + 0.5*k2, t + 0.5*h)
    k4 = h*model(x + k3, t + h)

    salida = x + (1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)
    
    return (salida)

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

paso_d = np.zeros(DAYS)
solucion = np.empty_like(z)
V_H = np.empty_like(paso_d)
egg_d = np.empty_like(paso_d)
egg_w = np.empty_like(paso_d)
larv  = np.empty_like(paso_d)
pupa  = np.empty_like(paso_d)
mosco = np.empty_like(paso_d)
aedes = np.empty_like(paso_d)
vec_s = np.empty_like(paso_d)
host_i = np.empty_like(paso_d)
host_s = np.empty_like(paso_d)
parametro   = np.zeros_like(paso_d)

egg_d[0] = z[0]#egg_wet(Rain[0])#v[0]/poblacion #mu_Dry*
egg_w[0] = z[1]
larv[0]  = z[2]
pupa[0]  = z[3]
mosco[0] = z[4]
aedes[0] = z[5]
vec_s[0] = z[6]
host_i[0] = 0.
host_s[0] = z[9]
parametro[0] = 0.

GT = np.empty_like(paso_d)

CI=4
control = np.zeros(DAYS)
control1=0

"""
for k in range(1,DAYS,1):
    casos_importados = señal(CI, k, DAYS)
    for i in range(0,DAYS-1,1):
        h=1.
        solucion = RK4(i,h,z)
        z = solucion
        #print(z[11])
        GT[i] = bite_rate * theta(Tmean[i]) * MIObv * z[8] * (z[9]/poblacion)

        control1 = control1 + GT[i]
        for j in range(13):
            if (z[j]<0.): z[j] = 0.
            else: z[j] = z[j]
        egg_d[i] = z[0]
        egg_w[i] = z[1]
        larv[i]  = z[2]
        pupa[i]  = z[3]
        mosco[i] = z[4]
        aedes[i] = z[5]
        vec_s[i] = z[6]
        host_s[i] = z[9]
        host_i[i] = z[11]
    control[k] = control1
    control1 = 0.

plt.plot(tiempo,control,label='CI=4')
#plt.title('2016-2017')
plt.ylabel('Nuevos Casos')
plt.grid()
plt.legend()
plt.show()
"""


################################### Para Control #######################################################

for i in range(0,DAYS-1,1):
    h=1.
    solucion = RK4(i,h,z)
    z = solucion
    #print(z[11])
    #GT[i] = bite_rate * theta(Tmean[i]) * MIObh * z[9] * (z[8]/poblacion)
    #if (Tmin[i] < 12.):
    #    GT[i] = 0.
    if (Tmin[i] < 12.):
        GT[i] = 0.
    else : GT[i] = bite_rate * theta(Tmean[i]) * MIObh * z[8] * (z[9]/poblacion)


    ### revisar el puto R0 ###
    #MUv=muerteV(Tmean[i])*MU_MOSQUITA_ADULTA
    #parametro[i]=((bite_rate*theta(Tmean[i]))**2)*MIObh*MIObv*(1./MUv)*(1.-np.exp(-(1/sigma(Tmean[i]))/MUv))*(host_s[i]/poblacion)*Remove_infect*(vec_s[i]/poblacion)
    #parametro[i]=((bite_rate*theta(Tmean[i]))**2)*MIObh*MIObv*(1./Remove_expose)*((MU_MOSQUITA_ADULTA*muerteV(Tmean[i])))*(1./Remove_infect)*(1./(1.+(0.1216*Tmean[i]*Tmean[i]-8.66*Tmean[i]+154.79)))*vec_s[i]/ALPHA
    #print(parametro[i])
    for j in range(13):
        if (z[j]<0.): z[j] = 0.
    else: z[j] = z[j]

    egg_d[i] = z[0]
    egg_w[i] = z[1]
    larv[i]  = z[2]
    pupa[i]  = z[3]
    mosco[i] = z[4]
    aedes[i] = z[5]
    vec_s[i] = z[6]
    host_s[i] = z[9]
    #parametro[i]=((bite_rate*theta(Tmean[i]))**2)*MIObh*MIObv*(1./Remove_expose)*((MU_MOSQUITA_ADULTA*muerteV(Tmean[i])))*(1./Remove_infect)*(1./(1.+(0.1216*Tmean[i]*Tmean[i]-8.66*Tmean[i]+154.79)))*vec_s[i]/ALPHA
    MUv=muerteV(Tmean[i])*MU_MOSQUITA_ADULTA
    parametro[i]=((bite_rate*theta(Tmean[i]))**2)*MIObh*MIObv*(1./MUv)*(1.-np.exp(-sigma(Tmean[i])/MUv))*(host_s[i]/poblacion)*Remove_infect*(vec_s[i]/poblacion)
    host_i[i] = z[11]

########################################################################################################
#plt.plot(tiempo,parametro)
#plt.plot(tiempo,GT)
#plt.show()

fig = plt.figure()
ax = plt.axes(xlim=(0,DAYS), ylim=(0,max(parametro)))
line, = ax.plot([],[])
line1, = ax.plot([],[])

def init():
    line.set_data([],[])
    return line,

def graficas(t):
    #a = np.zeros(np.size(x)) 
    #b = np.zeros(np.size(x))
    a = parametro [t]
    b= GT[t]
    return a,b

#uno,dos = graficas(6,tiempo)
#print(dos[0:10], uno[0:10])
x=[]
y=[]
z=[]

def animacion(i):
    #x = np.linspace(0,DAYS,DAYS)
    ys, zs = graficas(i)
    x.append(i)
    y.append(ys)
    z.append(zs)
    #print(i,z)
    line.set_data(x,y)
    line1.set_data(x,z)
    return line, line1

anim = animation.FuncAnimation(fig, animacion, init_func=init, frames= DAYS,
                               interval=20, blit = True)
anim.save('pu.gif', writer = animation.PillowWriter(fps=60))
plt.show()
