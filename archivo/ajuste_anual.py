import numpy as np
import pandas as pd
import optuna
pd.options.mode.chained_assignment = None # default='warn'
import scipy.special as sc
from selector_datos import seleccion, readinputdata
from semanas_epi2 import orden, orden_years2, casos_anual
from funciones import Fbar, egg_wet, C_Gillet, h_t_1, theta_T, rate_mx, rate_VE, muerte_V, calculo_EV, modelo
import matplotlib.pyplot as plt
#from sklearn.metrics import mean_squared_error



######################## FECHAS, INTERVALOS PARA LAS GRAFICAS ###################################
inicio = [1,'ene',2007]
final = [31,'dic',2010]
tiempo1 = np.arange('2007-01-02','2017-12-31',dtype='datetime64[D]') # para cualquier intervalo de tiempo
tiempo2 = np.arange('2009-01-01','2017-01-31',dtype='datetime64[M]') #para clinicos vs simulados (2009 al 2017)
tiempo3 = np.arange('2009-01-01','2010-01-01',dtype='datetime64[M]') #para clinicos vs simulados en un año especifico (desde 2009)
##################################################################################################

####################### CASOS CLINICOS, CASOS IMPORTADOS #################
casos_2009 = [30,246,112,2,6,1,0,0,0,0,0,0]
casos_2010 = [0,1,14,45,14,0,0,0,0,0,0,0]
casos_2011 = [0,1,1,0,0,0,0,0,0,0,0,0]
casos_2012 = [0,0,0,10,17,6,0,0,0,0,0,0]
casos_2013 = [0,10,39,42,20,0,0,0,0,0,0,0]
casos_2014 = [1,13,80,137,67,0,0,0,0,0,0,0]
casos_2015 = [0,0,0,0,1,0,0,0,0,0,0,0]
casos_2016 = [34,260,187,211,17,0,0,0,0,0,0,0]

d = {'casos_2009': casos_2009,
     'casos_2010': casos_2010,
     'casos_2011': casos_2011,
     'casos_2012': casos_2012,
     'casos_2013': casos_2013,
     'casos_2014': casos_2014,
     'casos_2015': casos_2015,
     'casos_2016': casos_2016}

df = pd.DataFrame(d) #TABLA DE CASOS CLINICOS

lala=readinputdata("datos_clinicos.txt")
casos_clinicos=lala[:,0]

importados = readinputdata('series_importados.txt')
casosImp = importados[:,7]  #SERIE DE CASOS IMPORTADOS
#importados = readinputdata('IMP_clinicos.txt')
#casosImp = importados[:,1] #SERIE DE CASOS IMPORTADOS
#################################################################################

############################# DATOS METEOROLOGICOS #############################
data=readinputdata('Oran_2001_nuevo.txt')
TMIN=data[:,0]
Tmean=data[:,2]
Rain=data[:,4]
HR=data[:,6]
hogares = 17633
poblacion = 75697

TMIN = seleccion(TMIN,inicio,final)
Tmean = seleccion(Tmean,inicio,final)
Rain = seleccion(Rain,inicio,final)
HR = seleccion(HR,inicio,final)
casosImp = seleccion(casosImp,inicio,final)
################################################################################

#################### PARAMETROS ###############################################
EGG_LIFE            = 120. #// vida de los huevos [dias]
EGG_LIFE_wet        = 90. #// vida de los huevos [dias]
mu_Dry              = 1./EGG_LIFE
mu_Wet              = 1./EGG_LIFE_wet
ALPHA               = 0.75#
Remove_infect       = 7.#//7. //dias 10
Remove_expose       = 5.#//7. //dias 829
MATAR_VECTORES      = 12.5#//9.5//12.5//temp
NO_INFECCION        = 15. #//15. //temp
NO_LATENCIA         = 16. #
MUERTE_ACUATICA     = 0.5
Temp_ACUATICA       = 10.
RATE_CASOS_IMP      = 1. #entero
#/*paramites aedes en days */
MU_MOSQUITO_JOVEN   = 1./2. #//0.091//1./10.//0.091 // 1./2.;//0.091;
MADURACION_MOSQUITO = 1./2. #//1./4.;//maduracion          
MU_MOSQUITA_ADULTA  = 1./10. #// vida^{-1} del vector en optimas condiciones (22 grados) 0.091; //
Rthres              = 12.5 #// 7.5 or 10.0 or 12.5
Hmax                = 24.#
kmmC                = 3.3E-6#//6.6E-5//3.9E-5//6.6E-5 //3.3E-6 // or 3.3E-6 or 6.6E-5 or 3.9E-5

DAYS = len(Tmean)
WEEKS = int(len(Tmean)/7) + 1
#print(DAYS,WEEKS)
###################################################################################

##############################################################################

####FUNCIONES A UTILIZAR ####FUNCIONES A UTILIZAR ####FUNCIONES A UTILIZAR

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def suv_exp(t,s,rate):
    
    salida = 0.
    tau    = t - s
    
    if ( t > 0. ):
        salida = np.exp(-rate*tau)
    
    return salida

####funcion para rk4
def rk4py2(f,t,h,X,args=()):
    n = len(X)
    salida = np.zeros(n)
    
    k1 = np.zeros(n)
    k2 = np.zeros(n)
    k3 = np.zeros(n)
    k4 = np.zeros(n)
    
    k1 = h*f(X, t,*args)#, Tmean, Rain, HR, TMIN, beta_day, hogares, poblacion, Kmax, Hmax, MADURACION_MOSQUITO, bite_rate, MIObv, NO_INFECCION, MU_MOSQUITA_ADULTA, MATAR_VECTORES, MU_MOSQUITO_JOVEN, casosImp, G_T, EV)
    k2 = h*f(X + 0.5*k1, t + 0.5*h,*args)#, Tmean, Rain, HR, TMIN, beta_day, hogares, poblacion, Kmax, Hmax, MADURACION_MOSQUITO, bite_rate, MIObv, NO_INFECCION, MU_MOSQUITA_ADULTA, MATAR_VECTORES, MU_MOSQUITO_JOVEN, casosImp, G_T, EV)
    k3 = h*f(X + 0.5*k2, t + 0.5*h,*args)#, Tmean, Rain, HR, TMIN, beta_day, hogares, poblacion, Kmax, Hmax, MADURACION_MOSQUITO, bite_rate, MIObv, NO_INFECCION, MU_MOSQUITA_ADULTA, MATAR_VECTORES, MU_MOSQUITO_JOVEN, casosImp, G_T, EV)
    k4 = h*f(X + k3, t + h,*args)#, Tmean, Rain, HR, TMIN, beta_day, hogares, poblacion, Kmax, Hmax, MADURACION_MOSQUITO, bite_rate, MIObv, NO_INFECCION, MU_MOSQUITA_ADULTA, MATAR_VECTORES, MU_MOSQUITO_JOVEN, casosImp, G_T, EV)

    salida = X + (1./6.)*( k1 + 2.*k2 + 2.*k3 + k4 )
    
    #for i in range(n):
        #salida[i] = round(salida[i],6)
    
    return salida

def rk4py(t,h,X):
    n = len(X)
    salida = np.zeros(n)
    
    k1 = np.zeros(n)
    k2 = np.zeros(n)
    k3 = np.zeros(n)
    k4 = np.zeros(n)
    
    k1 = h*modelo(X, t)#, Tmean, Rain, HR, TMIN, beta_day, hogares, poblacion, Kmax, Hmax, MADURACION_MOSQUITO, bite_rate, MIObv, NO_INFECCION, MU_MOSQUITA_ADULTA, MATAR_VECTORES, MU_MOSQUITO_JOVEN, casosImp, G_T, EV)
    k2 = h*modelo(X + 0.5*k1, t + 0.5*h)#, Tmean, Rain, HR, TMIN, beta_day, hogares, poblacion, Kmax, Hmax, MADURACION_MOSQUITO, bite_rate, MIObv, NO_INFECCION, MU_MOSQUITA_ADULTA, MATAR_VECTORES, MU_MOSQUITO_JOVEN, casosImp, G_T, EV)
    k3 = h*modelo(X + 0.5*k2, t + 0.5*h)#, Tmean, Rain, HR, TMIN, beta_day, hogares, poblacion, Kmax, Hmax, MADURACION_MOSQUITO, bite_rate, MIObv, NO_INFECCION, MU_MOSQUITA_ADULTA, MATAR_VECTORES, MU_MOSQUITO_JOVEN, casosImp, G_T, EV)
    k4 = h*modelo(X + k3, t + h)#, Tmean, Rain, HR, TMIN, beta_day, hogares, poblacion, Kmax, Hmax, MADURACION_MOSQUITO, bite_rate, MIObv, NO_INFECCION, MU_MOSQUITA_ADULTA, MATAR_VECTORES, MU_MOSQUITO_JOVEN, casosImp, G_T, EV)

    salida = X + (1./6.)*( k1 + 2.*k2 + 2.*k3 + k4 )
    
    #for i in range(n):
        #salida[i] = round(salida[i],6)
    
    return salida

def fun(parametros):
    ######################## FECHAS, INTERVALOS PARA LAS GRAFICAS ###################################
    #inicio = [1,'ene',2007]
    #final = [31,'dic',2009]
    #tiempo1 = np.arange('2007-01-02','2017-12-31',dtype='datetime64[D]') # para cualquier intervalo de tiempo
    #tiempo2 = np.arange('2009-01-01','2017-01-31',dtype='datetime64[M]') #para clinicos vs simulados (2009 al 2017)
    #tiempo3 = np.arange('2009-01-01','2010-01-01',dtype='datetime64[M]') #para clinicos vs simulados en un año especifico (desde 2009)
    ##################################################################################################

    #beta_day = 2.32     #AJUSTE ANUAL
    #Kmax = 190           #AJUSTE ANUAL
    MIObh         = 0.75#//0.0521*2. //0.0521 //1. //
    MIObv         = 0.75#//0.4867*2. //0.4867 //1. //
    bite_rate = 0.27 

    casos_2009 = [30,246,112,2,6,1,0,0,0,0,0,0]
    casos_2010 = [0,1,14,45,14,0,0,0,0,0,0,0]
    casos_2011 = [0,1,1,0,0,0,0,0,0,0,0,0]
    casos_2012 = [0,0,0,10,17,6,0,0,0,0,0,0]
    casos_2013 = [0,10,39,42,20,0,0,0,0,0,0,0]
    casos_2014 = [1,13,80,137,67,0,0,0,0,0,0,0]
    casos_2015 = [0,0,0,0,1,0,0,0,0,0,0,0]
    casos_2016 = [34,260,187,211,17,0,0,0,0,0,0,0]

    d = {'casos_2009': casos_2009,
         'casos_2010': casos_2010,
         'casos_2011': casos_2011,
         'casos_2012': casos_2012,
         'casos_2013': casos_2013,
         'casos_2014': casos_2014,
         'casos_2015': casos_2015,
         'casos_2016': casos_2016}
    
    df = pd.DataFrame(d)

    lala = readinputdata('datos_clinicos.txt')
    casos_clinicos = lala[:,0]

    importados = readinputdata('series_importados.txt')
    casosImp = importados[:,7]

    data=readinputdata('Oran_2001_nuevo.txt')
    TMIN=data[:,0]
    Tmean=data[:,2]
    Rain=data[:,4]
    HR=data[:,6]
    hogares = 17633
    poblacion = 75697
    TMIN = seleccion(TMIN,inicio,final)
    Tmean = seleccion(Tmean,inicio,final)
    Rain = seleccion(Rain,inicio,final)
    HR = seleccion(HR,inicio,final)
    casosImp = seleccion(casosImp,inicio,final)

    EGG_LIFE            = 120. #// vida de los huevos [dias]
    EGG_LIFE_wet        = 90. #// vida de los huevos [dias]
    mu_Dry              = 1./EGG_LIFE
    mu_Wet              = 1./EGG_LIFE_wet
    ALPHA               = 0.75#
    Remove_infect       = 7.#//7. //dias 10
    Remove_expose       = 5.#//7. //dias 829
    MATAR_VECTORES      = 12.5#//9.5//12.5//temp
    NO_INFECCION        = 15. #//15. //temp
    NO_LATENCIA         = 16. #
    MUERTE_ACUATICA     = 0.5
    Temp_ACUATICA       = 10.
    RATE_CASOS_IMP      = 1. #entero
    MU_MOSQUITO_JOVEN   = 1./2. #//0.091//1./10.//0.091 // 1./2.;//0.091;
    MADURACION_MOSQUITO = 1./2. #//1./4.;//maduracion          
    MU_MOSQUITA_ADULTA  = 1./10. #// vida^{-1} del vector en optimas condiciones (22 grados) 0.091; //
    Rthres              = 12.5 #// 7.5 or 10.0 or 12.5
    Hmax                = 24.#
    kmmC                = 3.3E-6#//6.6E-5//3.9E-5//6.6E-5 //3.3E-6 // or 3.3E-6 or 6.6E-5 or 3.9E-5
    DAYS = len(Tmean)
    WEEKS = int(len(Tmean)/7) + 1

    ED0  = 22876.
    EW0  = 102406
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

    dias = DAYS-1
    paso_d = np.zeros(dias)
    paso_w = np.zeros(WEEKS)
    solucion = np.empty_like(v)

    V_H = np.empty_like(paso_d)
    V_H_w = np.zeros_like(paso_w)
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
    G_T[0] = bite_rate*theta_T(Tmean[0]) * MIObv * v[8] * v[9]/poblacion
    G_TV[0] = bite_rate*theta_T(Tmean[0]) * MIObv * v[6] * v[11]/poblacion
    F_T[0] = bite_rate*theta_T(Tmean[0]) * MIObh * v[8] * v[6]/poblacion

    G_T_week = np.zeros(WEEKS)
    G_TV_week = np.zeros(WEEKS)
    contar = 0 
    week = 0

    pone = parametros[0] #beta day
    kmax = parametros[1] #bite rate
    #print(len(G_T))


    for t in range(1,dias):
        paso_d[int(t)] = t
        h = 1.
        G_T[t]	=  bite_rate*theta_T(Tmean[t])*MIObv * v[8] * v[9]/poblacion
        G_TV[t]	=  bite_rate*theta_T(Tmean[t])*MIObv * v[6] * v[11]/poblacion
        F_T[t] = bite_rate*theta_T(Tmean[0]) * MIObh * v[8] * v[6]/poblacion

        sigma_V     =	1./(1. + (0.1216*Tmean[int(t)]*Tmean[int(t)] - 8.66*Tmean[int(t)] + 154.79) )
        EV = calculo_EV(t,Tmean[t],v[7],G_T)
        H_t     = h_t_1(Rain[int(t)], Tmean[int(t)], HR[int(t)])
        #print(v)
        
        solucion     =  rk4py2(modelo,t,h,v,args=(EV,H_t,pone,kmax))
        v = solucion

        for q in range(13):
            if (v[q]<0.):
                v[q] = 0.

        V_H[t] = v[0]/poblacion

        egg_d[t] = v[0]/poblacion
        egg_w[t] = v[1]/poblacion
        larv[t] = v[2]/poblacion
        pupa[t] = v[3]/poblacion
        mosco[t] = v[4]/poblacion
        aedes[t] = v[5]/poblacion
        vec_s[t] = v[6]/poblacion
        vec_i[t] = v[8]/poblacion

        parametro[int(t)] =  ((bite_rate*theta_T(Tmean[t]))**2. )*MIObv*MIObh*(Remove_infect)*(1./(MU_MOSQUITA_ADULTA*muerte_V(Tmean[t])))*(1. - np.exp(-sigma_V*(1./(MU_MOSQUITA_ADULTA*muerte_V(Tmean[t])))))*vec_s[t]*ALPHA

        G_T_week[week] = G_T_week[week] + G_T[t]
        #G_TV_week[week] = G_TV_week[week] + G_TV_week[t]

        contar = contar + 1
        if (contar > 6):
            G_T_week[week] = G_T_week[week]
            #G_TV_week[week] = G_TV_week[week]
            week = week + 1
            contar = 0
        
        host_i[t] = v[11]
    
    casos_year = casos_anual(G_T_week,final[2])
    return casos_year

c= 'casos_nnnn'
eleccion = c.replace('nnnn',str(final[2]))
casos_clinic = np.array(df[eleccion])

def objetive(trial):
    x = trial.suggest_float("x",1,2.8)
    y = trial.suggest_float("y", 1, 3000)
    parametros = [x,y]
    y_pred = fun(parametros)
    y_val =  casos_clinic
    error = (np.square(y_val - y_pred)).mean()
    return error


study = optuna.create_study(direction = 'minimize')
study.optimize(objetive, n_trials = 2000)

best_params = study.best_params

found_x = best_params["x"]
found_y = best_params["y"]

print("FOUND beta_day:{}, kmax: {}".format(found_x,found_y))
