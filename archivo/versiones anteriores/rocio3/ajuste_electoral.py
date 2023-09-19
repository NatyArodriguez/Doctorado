import numpy as np
import pandas as pd
import optuna
import scipy.special as sc
from selector_datos import seleccion, readinputdata
from semanas_epi2 import orden, orden_years2, casos_anual, casos_temporada
from funciones import Fbar, egg_wet, C_Gillet, h_t_1, theta_T, rate_mx, rate_VE, muerte_V, calculo_EV, modelo
import matplotlib.pyplot as plt


######################## FECHAS, INTERVALOS PARA LAS GRAFICAS ###################################
inicio = [1,'ene',2007]
final = [31,'dic',2017]
tiempo1 = np.arange('2007-01-02','2017-12-31',dtype='datetime64[D]') # para cualquier intervalo de tiempo
tiempo2 = np.arange('2009-01-01','2017-01-31',dtype='datetime64[M]') #para clinicos vs simulados (2009 al 2017)
tiempo3 = np.arange('2009-01-01','2010-01-01',dtype='datetime64[M]') #para clinicos vs simulados en un año especifico (desde 2009)
##################################################################################################

####################### CASOS CLINICOS, CASOS IMPORTADOS #################
casos_2009 = [0,0,0,0,0,0,30,246,112,2,6,1]
casos_2010 = [0,0,0,0,0,0,0,1,14,45,14,0]
casos_2011 = [0,0,0,0,0,0,0,1,1,0,0,0]
casos_2012 = [0,0,0,0,0,0,0,0,0,10,17,6]
casos_2013 = [0,0,0,0,0,0,0,10,39,42,20,0]
casos_2014 = [0,0,0,0,0,0,1,13,80,137,67,0]
casos_2015 = [0,0,0,0,0,0,0,0,0,0,1,0]
casos_2016 = [0,0,0,0,0,0,34,260,187,211,17,0]
casos_confirmados = np.concatenate((casos_2009,casos_2010,casos_2011,casos_2012,casos_2013,casos_2014,casos_2015,casos_2016))

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
#casosImp = importados[:,0] #SERIE DE CASOS IMPORTADOS
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


def fun(parametros):
    MIObh         = 0.75#//0.0521*2. //0.0521 //1. //
    MIObv         = 0.75#//0.4867*2. //0.4867 //1. //
    bite_rate = 0.27 

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
    #Remove_infect       = 7.#//7. //dias 10
    #Remove_expose       = 5.#//7. //dias 829
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

    b7 = b8 = b9 = parametros[0] 
    b10 = parametros[1]
    b11 = parametros[2]
    b12 = parametros[3]
    b13 = parametros[4]
    b14 = parametros[5]
    b15 = parametros[6]
    b16 = b17 = parametros[7]
    
    k7 = k8 = k9 = parametros[8]
    k10 = parametros[9]
    k11 = parametros[10]
    k12 = parametros[11]
    k13 = parametros[12]
    k14 = parametros[13]
    k15 = parametros[14]
    k16= k17 = parametros[15]

    e7 = e8 = e9 = parametros[16]
    e10 = parametros[17]
    e11 = parametros[18]
    e12 = parametros[19]
    e13 = parametros[20]
    e14 = parametros[21]
    e15 = parametros[22]
    e16 = e17 = parametros[23]

    i7 = i8 = i9 = parametros[24]
    i10 = parametros[25]
    i11 = parametros[26]
    i12 = parametros[27]
    i13 = parametros[28]
    i14 = parametros[29]
    i15 = parametros[30]
    i16 = i17 = parametros[31]
    

    julio = [180,608,911,1276,1641,2007,2372,2737,3102,3468]
    BETA = [b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17]
    KMAX = [k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17]
    R_EXPOSE = [e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17]
    R_INFECT = [i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17]
    #BETA = [1.47  , 1.47  , 1.47  , 1.018 , 2.72   , 1.498, 1.96     , 1.39  , 2.15, pone]
    #KMAX = [231.52, 231.52, 231.52, 567.89, 1181.72, 17.915, 1981.005, 183.61, 1681.09, kmax]
    valor = 0


    for t in range(1,dias):
        paso_d[int(t)] = t
        h = 1.

        beta_day = BETA[valor]
        Kmax = KMAX[valor]
        Remove_expose = R_EXPOSE[valor]
        Remove_infect = R_INFECT[valor]

        G_T[t]	=  bite_rate*theta_T(Tmean[t])*MIObv * v[8] * v[9]/poblacion
        G_TV[t]	=  bite_rate*theta_T(Tmean[t])*MIObv * v[6] * v[11]/poblacion
        F_T[t] = bite_rate*theta_T(Tmean[0]) * MIObh * v[8] * v[6]/poblacion

        sigma_V     =	1./(1. + (0.1216*Tmean[int(t)]*Tmean[int(t)] - 8.66*Tmean[int(t)] + 154.79) )
        EV = calculo_EV(t,Tmean[t],v[7],G_T)
        H_t     = h_t_1(Rain[int(t)], Tmean[int(t)], HR[int(t)])
        #print(v)
        
        solucion     =  rk4py2(modelo,t,h,v,args=(EV,H_t,beta_day,Kmax,Remove_expose,Remove_infect))
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
        if t in julio:
            valor = valor + 1
        beta_day = BETA[valor]
        Kmax = KMAX[valor]

    c1 = casos_temporada(G_T_week,2009)
    c2 = casos_temporada(G_T_week,2010)
    c3 = casos_temporada(G_T_week,2011)
    c4 = casos_temporada(G_T_week,2012)
    c5 = casos_temporada(G_T_week,2013)
    c6 = casos_temporada(G_T_week,2014)
    c7 = casos_temporada(G_T_week,2015)
    c8 = casos_temporada(G_T_week,2016)
    casos_year = np.concatenate((c1,c2,c3,c4,c5,c6,c7,c8))
    #print(casos_year)
    return casos_year

#c= 'casos_nnnn'
#eleccion = c.replace('nnnn',str(final[2]))
#casos_clinic = np.array(df[eleccion])
#print(casos_confirmados)

def objetive(trial):
    x1 = trial.suggest_float("x1",0,2.8) #beta day
    x2 = trial.suggest_float("x2",0,2.8)
    x3 = trial.suggest_float("x3",0,0.1)
    x4 = trial.suggest_float("x4",0,2.8)
    x5 = trial.suggest_float("x5",0,2.8)
    x6 = trial.suggest_float("x6",0,2.8)
    x7 = trial.suggest_float("x7",0,0.1)
    x8 = trial.suggest_float("x8",0,2.8)

    y1 = trial.suggest_float("y1", 1, 3000) #kmax
    y2 = trial.suggest_float("y2", 1, 3000)
    y3 = trial.suggest_float("y3", 1, 100)
    y4 = trial.suggest_float("y4", 1, 3000)
    y5 = trial.suggest_float("y5", 1, 3000)
    y6 = trial.suggest_float("y6", 1, 3000)
    y7 = trial.suggest_float("y7", 1, 100)
    y8 = trial.suggest_float("y8", 1, 3000)

    z1 = trial.suggest_float("z1", 3, 10)
    z2 = trial.suggest_float("z2", 3, 10)
    z3 = trial.suggest_float("z3", 3, 10)
    z4 = trial.suggest_float("z4", 3, 10)
    z5 = trial.suggest_float("z5", 3, 10)
    z6 = trial.suggest_float("z6", 3, 10)
    z7 = trial.suggest_float("z7", 3, 10)
    z8 = trial.suggest_float("z8", 3, 10)

    w1 = trial.suggest_float("w1", 3, 10)
    w2 = trial.suggest_float("w2", 3, 10)
    w3 = trial.suggest_float("w3", 3, 10)
    w4 = trial.suggest_float("w4", 3, 10)
    w5 = trial.suggest_float("w5", 3, 10)
    w6 = trial.suggest_float("w6", 3, 10)
    w7 = trial.suggest_float("w7", 3, 10)
    w8 = trial.suggest_float("w8", 3, 10)

    parametros = [x1,x2,x3,x4,x5,x6,x7,x8,y1,y2,y3,y4,y5,y6,y7,y8,z1,z2,z3,z4,z5,z6,z7,z8,w1,w2,w3,w4,w5,w6,w7,w8]
    y_pred = fun(parametros)
    y_val =  casos_confirmados
    error = (np.square(y_val - y_pred)).mean()
    arch = open("guardar00.txt","r+")
    arch.read()
    print("%f"%error,end ="\t",file=arch)
    print(*parametros,sep ="\t",end ="\n",file=arch)
    arch.close()
    return error


study = optuna.create_study(direction = 'minimize')
study.optimize(objetive, n_trials = 3000)

best_params = study.best_params

found_x1 = best_params["x1"]
found_x2 = best_params["x2"]
found_x3 = best_params["x3"]
found_x4 = best_params["x4"]
found_x5 = best_params["x5"]
found_x6 = best_params["x6"]
found_x7 = best_params["x7"]
found_x8 = best_params["x8"]
found_y1 = best_params["y1"]
found_y2 = best_params["y2"]
found_y3 = best_params["y3"]
found_y4 = best_params["y4"]
found_y5 = best_params["y5"]
found_y6 = best_params["y6"]
found_y7 = best_params["y7"]
found_y8 = best_params["y8"]
found_z1 = best_params["z1"]
found_z2 = best_params["z2"]
found_z3 = best_params["z3"]
found_z4 = best_params["z4"]
found_z5 = best_params["z5"]
found_z6 = best_params["z6"]
found_z7 = best_params["z7"]
found_z8 = best_params["z8"]
found_w1 = best_params["w1"]
found_w2 = best_params["w2"]
found_w3 = best_params["w3"]
found_w4 = best_params["w4"]
found_w5 = best_params["w5"]
found_w6 = best_params["w6"]
found_w7 = best_params["w7"]
found_w8 = best_params["w8"]

print("FOUND b9:{} ,b10:{}, b11:{}, b12:{},b13:{},b14:{},b15:{},b16:{},k9:{},k10:{},k11:{},k12:{},k13:{},k14:{},k15:{},k16:{},e9:{} ,e10:{},e11:{},e12:{},e13:{},e14:{},e15:{},e16:{},i9:{},i10:{},i11:{},i12:{},i13:{},i14:{},i15:{},i16:{}".
      format(found_x1,
             found_x2,
             found_x3,
             found_x4,
             found_x5,
             found_x6,
             found_x7,
             found_x8,
             found_y1,
             found_y2,
             found_y3,
             found_y4,
             found_y5,
             found_y6,
             found_y7,
             found_y8,
             found_z1,
             found_z2,
             found_z3,
             found_z4,
             found_z5,
             found_z6,
             found_z7,
             found_z8,
             found_w1,
             found_w2,
             found_w3,
             found_w4,
             found_w5,
             found_w6,
             found_w7,
             found_w8))
