import numpy as np
import pandas as pd
import scipy.special as sc
from funciones import  casos_temporada
import matplotlib.pyplot as plt
import funciones as fn

######################## FECHAS, INTERVALOS PARA LAS GRAFICAS ###################################
dias = np.arange('2001-01-01','2018-01-01',dtype='datetime64[D]') #6209
inicio = '2007-01-01' #Este se mantiene fijo
fin = '2017-12-31' # Solo cambiar el año
temporada = pd.date_range(start=inicio, end=fin, freq='D')

oran = pd.read_table('ORAN.txt',sep='\t')
oran.index = dias

serie = pd.read_table('series_importados.txt',sep ='\t')
serie.index = dias

rango = oran.loc[temporada]
importados = serie.loc[temporada]

TMIN = rango['tmin']
Tmean = rango['tmean']
Rain = rango['rain'] 
HR = rango['hr']
casosImp = importados['s7']
##################################################################################################

################ PARAMETROS AJUSTADOS ##################################
b7 = b8 = b9 = 2.40
b10 = 1.69
b11 = 1.54
b12 = 1.80
b13 = 1.68
b14 = 2.1
b15 = 1.65
b16 = 2.52
b17 = b16

k7 = k8 = k9 = 306.03
k10 = 449.96
k11 = 500.89
k12 = 800.02
k13 = 819.99
k14 = 700.00
k15 = 359.99
k16 = 600.06
k17 = k16

# elemento = [ 6-7, 7-8, 8-9, 9-10, 10-11, 11-12, 12-13, 13-14, 14-15, 15-16]
Beta_day = [b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17]
kmax  = [k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17]
#######################################################################

####################### CASOS CLINICOS, CASOS IMPORTADOS #################
casos_2009 = [0,0,0,0,0,0,30,246,112,2,6,1]
casos_2010 = [0,0,0,0,0,0,0,1,14,45,14,0]
casos_2011 = [0,0,0,0,0,0,0,1,1,0,0,0]
casos_2012 = [0,0,0,0,0,0,0,0,0,10,17,6]
casos_2013 = [0,0,0,0,0,0,0,10,39,42,20,0]
casos_2014 = [0,0,0,0,0,0,1,13,80,137,67,0]
casos_2015 = [0,0,0,0,0,0,0,0,0,0,1,0]
casos_2016 = [0,0,0,0,0,0,34,260,187,211,17,0]

d = {'casos_2009': casos_2009,
     'casos_2010': casos_2010,
     'casos_2011': casos_2011,
     'casos_2012': casos_2012,
     'casos_2013': casos_2013,
     'casos_2014': casos_2014,
     'casos_2015': casos_2015,
     'casos_2016': casos_2016}

df = pd.DataFrame(d) #TABLA DE CASOS CLINICOS

#################### PARAMETROS ###############################################
MIObh         = fn.MIObh#//0.0521*2. //0.0521 //1. //
MIObv         = fn.MIObv#//0.4867*2. //0.4867 //1. //
bite_rate = fn.bite_rate
poblacion = fn.poblacion
hogares = fn.hogares
ALPHA     = fn.ALPHA# 
Remove_infect       = fn.Remove_infect#//7. //dias 10
Remove_expose       = fn.Remove_expose#//7. //dias 829
MU_MOSQUITA_ADULTA  = fn.MU_MOSQUITA_ADULTA    
EGG_LIFE            = fn.EGG_LIFE #// vida de los huevos [dias]
EGG_LIFE_wet        = fn.EGG_LIFE_wet #// vida de los huevos [dias]
mu_Dry              = 1./EGG_LIFE
mu_Wet              = 1./EGG_LIFE_wet
MATAR_VECTORES      = fn.MATAR_VECTORES#//9.5//12.5//temp
NO_INFECCION        = fn.NO_INFECCION #//15. //temp
NO_LATENCIA         = fn.NO_LATENCIA #
MUERTE_ACUATICA     = fn.MUERTE_ACUATICA
Temp_ACUATICA       = fn.Temp_ACUATICA
RATE_CASOS_IMP      = fn.RATE_CASOS_IMP #entero
MU_MOSQUITO_JOVEN   = fn.MU_MOSQUITO_JOVEN #//0.091//1./10.//0.091 // 1./2.;//0.091;
MADURACION_MOSQUITO = fn.MADURACION_MOSQUITO #//1./4.;//maduracion          
Rthres              = fn.Rthres #// 7.5 or 10.0 or 12.5
Hmax                = fn.Hmax#
kmmC                = fn.kmmC#//6.6E-5//3.9E-5//6.6E-5 //3.3E-6 // or 3.3E-6 or 6.6E-5 or 3.9E-5

DAYS = len(Tmean)
WEEKS = int(len(Tmean)/7) + 1
###################################################################################


################### Condiciones iniciales ################################
ED0  = 22876.
EW0  = 102406.
L0   = 24962.
P0   = 2003.
M0   = 28836.
V0   = 0.
V_S0 = 0.
V_E0 = 0.
V_I0 = 0.
H_S0 = ALPHA*poblacion
H_E0 = 0.
H_I0 = 0.
H_R0 = ALPHA*poblacion - H_S0
H_t  = 24.

#pasando todo a arreglo

v = np.zeros(13)
v[0]  = ED0
v[1]  = EW0
v[2]  = L0
v[3]  = P0
v[4]  = M0
v[5]  = V0
v[6]  = V_S0
v[7]  = V_E0
v[8]  = V_I0
v[9]  = H_S0
v[10] = H_E0
v[11] = H_I0
v[12] = H_R0
###############################################################################

###################### Ahora a resolver la ODE #######################
dias = DAYS-1
paso_d = np.zeros(dias)
paso_w = np.zeros(WEEKS)

solucion = np.empty_like(v)

V_H = np.empty_like(paso_d)
V_H_w = np.zeros_like(paso_w)
####Soluciones
egg_d = np.empty_like(paso_d)
egg_w = np.empty_like(paso_d)
larv  = np.empty_like(paso_d)
pupa  = np.empty_like(paso_d)
mosco = np.empty_like(paso_d)
aedes = np.empty_like(paso_d)
vec_s = np.empty_like(paso_d)
vec_i = np.empty_like(paso_d)
host_i = np.empty_like(paso_d)
parametro   = np.zeros_like(paso_d)
parametro_w = np.zeros_like(paso_w)

egg_d[0] = v[0]/poblacion#egg_wet(Rain[0])#v[0]/poblacion #mu_Dry*
egg_w[0] = v[1]/poblacion
larv[0]  = v[2]/poblacion
pupa[0]  = v[3]/poblacion
mosco[0] = v[4]/poblacion
aedes[0] = v[4]/poblacion
vec_s[0] = v[4]/poblacion
vec_i[0] = v[8]/poblacion
host_i[0]= 0.
parametro[0] = 0.

G_T = np.empty_like(paso_d)
G_TV = np.empty_like(paso_d)
F_T = np.empty_like(paso_d)
G_T[0] = bite_rate*fn.theta_T(Tmean[0]) * MIObv * v[8] * v[9]/poblacion
G_TV[0] = bite_rate*fn.theta_T(Tmean[0]) * MIObv * v[6] * v[11]/poblacion
F_T[0] = bite_rate*fn.theta_T(Tmean[0]) * MIObh * v[8] * v[6]/poblacion

G_T_week = np.zeros(WEEKS)
G_TV_week = np.zeros(WEEKS)

julio = [180,546,911,1276,1641,2007,2372,2737,3102,3468]

contar = 0 
valor = 0
week = 0

for t in range(1,dias):
    paso_d[int(t)]   = t
    h = 1.
    beta_day = Beta_day[valor]
    Kmax = kmax[valor]
    #print('inicio',t, beta_day,Kmax)
    #print(t,Kmax,beta_day)
        
    G_T[t]	=  bite_rate*fn.theta_T(Tmean[t])*MIObv * v[8] * v[9]/poblacion
    G_TV[t]	=  bite_rate*fn.theta_T(Tmean[t])*MIObv * v[6] * v[11]/poblacion
    F_T[t] = bite_rate*fn.theta_T(Tmean[0]) * MIObh * v[8] * v[6]/poblacion

    sigma_V     =	1./(1. + (0.1216*Tmean[int(t)]*Tmean[int(t)] - 8.66*Tmean[int(t)] + 154.79) )
        
    EV = sigma_V*v[7]#calculo_EV(t,Tmean[t],v[7],G_T)

    H_t     = fn.hume(H_t,Rain[int(t)], Tmean[int(t)], HR[int(t)])
        
    solucion     =  fn.runge(fn.modelo,t,h,v,args=(EV,H_t,Tmean,TMIN,Rain,casosImp,beta_day,Kmax))
    v = solucion
    #print(v[12])
    for q in range(13):
        if (v[q]< 0.):
            v[q] = 0.
        
    V_H[t] = v[0]/poblacion
        
    egg_d[t] = v[0]/poblacion#egg_wet(Rain[int(t)])#v[0]/poblacion
    egg_w[t] = v[1]/poblacion
    larv[t]  = v[2]/poblacion
    pupa[t]  = v[3]/poblacion
    mosco[t] = v[4]/poblacion
    aedes[t] = v[5]/poblacion
    vec_s[t] = v[6]/poblacion
    vec_i[t] = v[8]/poblacion 
    
    gamma = 1./Remove_infect  
    
    #parametro[int(t)] =  ((bite_rate*theta_T(Tmean[t]))**2. )*MIObv*MIObh*(Remove_infect)*(1./(MU_MOSQUITA_ADULTA*muerte_V(Tmean[t])))*(1. - np.exp(-sigma_V*(1./(MU_MOSQUITA_ADULTA*muerte_V(Tmean[t])))))*vec_s[t]*ALPHA
    parametro[int(t)] = np.sqrt( (sigma_V/gamma) * ((bite_rate*bite_rate*fn.theta_T(Tmean[t])*fn.theta_T(Tmean[t])*MIObh*MIObv)/(fn.muerte_V(Tmean[t])*MU_MOSQUITA_ADULTA*(sigma_V + fn.muerte_V(Tmean[t])*MU_MOSQUITA_ADULTA))) * vec_s[t] * ALPHA )
    
    
    G_T_week[week] = G_T_week[week] + G_T[t]
    G_TV_week[week] = G_TV_week[week] + G_TV[t]    
    
    contar = contar + 1
    if (contar > 6):
        G_T_week[week] = G_T_week[week]
        G_TV_week[week] = G_TV_week[week]
        week = week +1
        contar = 0.
    #parametro_w[week] = parametro_w[week] +  parametro[t]
    host_i[t] = v[11]

    if t in julio:
        valor = valor + 1
        #print(t,valor)
    beta_day = Beta_day[valor]
    Kmax = kmax[valor]


###################### TABLA CLINICOS VS SIMULADOS ###############################
''' Casos acumulados '''

years = [2009,2010,2011,2012,2013,2014,2015,2016]
clinicos_anuales = [397, 74, 2, 33, 111, 298, 1, 709]
c1 = casos_temporada(G_T_week,2009)
c2 = casos_temporada(G_T_week,2010)
c3 = casos_temporada(G_T_week,2011)
c4 = casos_temporada(G_T_week,2012)
c5 = casos_temporada(G_T_week,2013)
c6 = casos_temporada(G_T_week,2014)
c7 = casos_temporada(G_T_week,2015)
c8 = casos_temporada(G_T_week,2016)

generados_anuales = [sum(c1),sum(c2),sum(c3),sum(c4),sum(c5),sum(c6),sum(c7),sum(c8)]

d = {'clinicos':clinicos_anuales, 'simulacion':generados_anuales}
df2 = pd.DataFrame(d,index = years)

print(df2)

df2.plot(kind='bar')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.title('Histograma de Datos por Año')
plt.xticks(rotation=45)
##################################################################################


############################ GRAFICA DE LOS RESULTADOS #########################
periodo=[inicio,fin]
WEEKS = np.linspace(0,WEEKS,WEEKS)

casos_confirmados = np.concatenate((casos_2009,casos_2010,casos_2011,casos_2012,casos_2013,casos_2014,casos_2015,casos_2016))
casos_year = np.concatenate((c1,c2,c3,c4,c5,c6,c7,c8))

movil = fn.moving_average(parametro,7)
rango = np.arange('2007-01-07','2017-12-31',dtype='datetime64[D]')

tiempo1 = np.arange('2007-01-01','2017-12-31',dtype='datetime64[D]') # para cualquier intervalo de tiempo
tiempo2 = np.arange('2009-01-01','2017-01-31',dtype='datetime64[M]') #para clinicos vs simulados (2009 al 2017)

plt.figure(figsize=(10,6))
#plt.plot(tiempo1,parametro,label='R_0') #grafica r_0
#plt.plot(rango,movil,label='R_0 movil') #grafiva r_0 movil
plt.plot(tiempo2,casos_confirmados, 'r', label='clinicos')
plt.plot(tiempo2,casos_year, '--g', label ='simulacion')
plt.title(periodo)
plt.grid()
plt.legend()
plt.show()
