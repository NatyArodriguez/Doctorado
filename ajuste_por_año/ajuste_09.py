import numpy as np
import pandas as pd
import optuna
import funciones as fn

#################### READ ME ####################
# Agregar el mejor valor encontrado a los arreglos BETA y KMAX, lineas 117 a 122
# Cambiar el nombre del archivo en la linea 196
########################################################

############################# DATOS METEOROLOGICOS #############################
dias = np.arange('2001-01-01','2018-01-01',dtype='datetime64[D]') #6209
inicio = '2007-01-01' #Este se mantiene fijo
fin = '2009-07-01' # Solo cambiar el año
final = 2009 #cambiar el año
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


def fun(parametros):
    MIObh         = fn.MIObh
    MIObv         = fn.MIObv
    bite_rate = fn.bite_rate
    ALPHA               = fn.ALPHA
    Remove_infect       = fn.Remove_infect
    Remove_expose    = fn.Remove_expose
    MU_MOSQUITA_ADULTA  = fn.MU_MOSQUITA_ADULTA
    poblacion = fn.poblacion
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
    G_T[0] = bite_rate*fn.theta_T(Tmean[0]) * MIObv * v[8] * v[9]/poblacion
    G_TV[0] = bite_rate*fn.theta_T(Tmean[0]) * MIObv * v[6] * v[11]/poblacion
    F_T[0] = bite_rate*fn.theta_T(Tmean[0]) * MIObh * v[8] * v[6]/poblacion

    G_T_week = np.zeros(WEEKS)
    G_TV_week = np.zeros(WEEKS)
    contar = 0 
    week = 0

    julio = [181,609,912,1277,1642,2008,2373,2738,3103,3469]

    b7 = b8 = b9 = parametros[0]
    #b10 = 1.69
    #b11 = 1.54
    #b12 = 1.80
    #b13 = 1.68
    #b14 = 2.10
    #b15 = 1.65
    #b16 = parametros[0]
    #b10 =parametros[0] #beta day
    k7 = k8 = k9 = parametros[1]
    #k10 = 449.96
    #k11 = 500.89
    #k12 = 800.02
    #k13 = 819.99
    #k14 = 700.00
    #k15 = 359.99
    #k16 = parametros[1]
    #k10 = parametros[1] #bite rate
    BETA = [b7,b8,b9]
    KMAX = [k7,k8,k9]

    valor = 0

    for t in range(1,dias):
        #print(valor)
        paso_d[int(t)] = t
        h = 1.

        beta_day = BETA[valor]
        Kmax = KMAX[valor]

        G_T[t]	=  bite_rate*fn.theta_T(Tmean[t])* MIObv * v[8] * v[9]/poblacion
        G_TV[t]	=  bite_rate*fn.theta_T(Tmean[t])* MIObv * v[6] * v[11]/poblacion
        F_T[t] = bite_rate*fn.theta_T(Tmean[0]) * MIObh * v[8] * v[6]/poblacion

        sigma_V     =	1./(1. + (0.1216*Tmean[int(t)]*Tmean[int(t)] - 8.66*Tmean[int(t)] + 154.79) )

        EV = sigma_V*v[7]
        H_t     = fn.hume(H_t,Rain[int(t)], Tmean[int(t)], HR[int(t)])
        #print(v)
        
        solucion     =  fn.runge(fn.modelo,t,h,v,args=(EV,H_t,Tmean,TMIN,Rain,casosImp,beta_day,Kmax))
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

        parametro[int(t)] =  ((bite_rate*fn.theta_T(Tmean[t]))**2. )*MIObv*MIObh*(Remove_infect)*(1./(MU_MOSQUITA_ADULTA*fn.muerte_V(Tmean[t])))*(1. - np.exp(-sigma_V*(1./(MU_MOSQUITA_ADULTA*fn.muerte_V(Tmean[t])))))*vec_s[t]*ALPHA

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
            #print(t,valor)
            
        beta_day = BETA[valor]
        Kmax = KMAX[valor]
    casos_year = fn.casos_temporada(G_T_week,final)
    return casos_year

c= 'casos_nnnn'
eleccion = c.replace('nnnn',str(final))
casos_clinic = np.array(fn.df[eleccion])

def objetive(trial):
    x = trial.suggest_float("x",0.1,2.8) #beta day
    y = trial.suggest_float("y", 100, 1000) #kmax
    parametros = [x,y]
    y_pred = fun(parametros)
    y_val =  casos_clinic
    error = (np.square(y_val - y_pred)).mean()
    #arch = open('temp16.txt','r+')
    #arch.read()
    #print("%f"%error,end = '\t',file=arch)
    #print(*parametros,sep='\t',end='\n',file=arch)
    #arch.close()
    return error


study = optuna.create_study(direction = 'minimize')
study.optimize(objetive, n_trials = 100)

best_params = study.best_params

found_x = best_params["x"]
found_y = best_params["y"]

print("FOUND beta_day:{}, kmax: {}".format(found_x,found_y))

arch = open('parametro.txt','r+')
arch.read()
valores = []
for key, value in study.best_trial.params.items():
    valores.append(value)
print(*valores,sep='\t',end='\n',file=arch)
arch.close()