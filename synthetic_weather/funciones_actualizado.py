import numpy as np
import pandas as pd
import scipy.special as sc
import random
import math

####################### Tablas de oran #######################
df_oran = pd.read_table('ORAN_2001_2022.txt', sep='\t') #--> tmin tmax tmean rain hr
days_oran = pd.date_range(start='2001-01-01', end='2022-12-31', freq='D')
df_oran.index = days_oran

df_medio = pd.read_table('Oran_2001_2017_medio.txt', sep='\t')#--> tmin tmax tmean tmedian rain hrm hr
days_medio = pd.date_range(start='2001-01-01', end='2017-12-31', freq='D')
df_medio.index = days_medio

###################### FUNCIONES Y PARAMETROS ######################

################ PARAMETROS A AJUSTAR ##################################
MIObh         = 0.75#//0.0521*2. //0.0521 //1. //
MIObv         = 0.75#//0.4867*2. //0.4867 //1. //
bite_rate = 0.27      #0.12 // 0.18 // 0.16 // 0.25 // 0.16  #0.21  # 0.21
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
H_t  = 24.
hogares = 17633
poblacion = 75697
#################################################################################

################# Casos clinicos ############################
casos_2009 = [0,0,0,0,0,0,30,246,112,2,6,1]
casos_2010 = [0,0,0,0,0,0,0,1,14,45,14,0]
casos_2011 = [0,0,0,0,0,0,0,1,1,0,0,0]
casos_2012 = [0,0,0,0,0,0,0,0,0,10,17,6]
casos_2013 = [0,0,0,0,0,0,0,10,39,42,20,0]
casos_2014 = [0,0,0,0,0,0,1,13,80,137,67,0]
casos_2015 = [0,0,0,0,0,0,0,0,0,0,1,0]
casos_2016 = [0,0,0,0,0,0,34,260,187,211,17,0]

casos_confirmados = np.concatenate((casos_2009,casos_2010,casos_2011,casos_2012,
                                    casos_2013,casos_2014,casos_2015,casos_2016))

d = {'casos_2009': casos_2009,
     'casos_2010': casos_2010,
     'casos_2011': casos_2011,
     'casos_2012': casos_2012,
     'casos_2013': casos_2013,
     'casos_2014': casos_2014,
     'casos_2015': casos_2015,
     'casos_2016': casos_2016}

df_casos_confirmados = pd.DataFrame(d)
###############################################################################

####################### ALPHA ###############################
al_hist = {
    #'month':
    #    [1, 2, 3, 4, 5, 6,
    #     7, 8, 9, 10, 11, 12],
    'alpha':
        #[0.38, 0.34, 0.46, 0.37, 0.53, 0.41,
        # 0.38, 0.26, 0.59, 0.58, 0.53, 0.43],
        [0.38, 0.34, 0.46, 0.37, 0.62, 0.41,
         0.45, 0.58, 0.63, 0.58, 0.53, 0.43],
    'std':
        #[0.11, 0.08, 0.12, 0.15, 0.19, 0.28,
        # 0.24, 0.35, 0.33, 0.27, 0.14, 0.17]
        [0.11, 0.08, 0.12, 0.15, 0.25, 0.28,
         0.30, 0.26, 0.25, 0.27, 0.14, 0.17]
        }

SEM = [0.01, 0.01, 0.01, 0.01, 0.02, 0.03,
       0.03, 0.12, 0.05, 0.02, 0.01, 0.02]

hystoric_al = pd.DataFrame(al_hist)
hystoric_al['alpha+std'] = hystoric_al['alpha'] + hystoric_al['std']
hystoric_al['alpha-std'] = hystoric_al['alpha'] - hystoric_al['std']
hystoric_al[hystoric_al < 0] = 0 #un alpha negativo no tiene sentido
################################################################################

####FUNCIONES A UTILIZAR ####FUNCIONES A UTILIZAR ####FUNCIONES A UTILIZAR

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def suv_exp(t,s,rate):
    
    salida = 0.
    tau    = t - s
    
    if ( t > 0. ):
        salida = np.exp(-rate*tau)
    
    return salida


def Fbar(t,s,k,theta):
    salida = 0.
    tau = t - s
    
    if (t > 0. ):
        
        salida = sc.gammaincc(k, tau/theta)
        
    return salida
    
##funciones 
#fR fraccion de ED -> EW
def egg_wet(rain):
    salida = 0.
    lluvia   = rain / Rthres 
    salida = 0.8*lluvia**5/(1. + lluvia**5)
    
    return salida

#modelo Gillet
def C_Gillet(Larva,KL):
    salida = 0.
    if (Larva < KL):
        salida = 1. - Larva/KL
    else:
        salida = 0.
    
    return salida
#H(t+1)
def hume(H_t,rain,Tm,Hum):
    
    salida = H_t + rain - kmmC*(100. - Hum)*(25. + Tm)*(25. + Tm)
    
    if ( salida < 0.):
        salida = 0.
    
    if (Hmax < salida): 
        salida = Hmax
    
    return salida
#actividad
def theta_T(Tm):
    salida = 0.
    
    if ( (11.7 < Tm) and (Tm < 32.7)) :
        salida = 0.1137*(-5.4 + 1.8*Tm - 0.2124*Tm*Tm + 0.01015*Tm*Tm*Tm - 0.0001515*Tm*Tm*Tm*Tm);
    
    return salida

#tasas de maduracion L,P,M
def rate_mx(Tm,DHA,DHH,T12):
    salida = 0.
    aux0 = (DHA/1.987207) * (1./298. - 1./(Tm + 273.15) )
    aux1 = (DHH/1.987207) * (1./T12 - 1./(Tm + 273.15) )
    salida = ( (Tm + 273.15)/298. ) * ( np.exp( aux0 )  / (1.+ np.exp(aux1)) )

    return salida
#tasa de latencia mosquitos
def rate_VE(Tm,k):
    salida = k/np.exp( -0.155*Tm + 6.5031 )
    if (k<=3):
        salida = k/np.exp( -0.1659*Tm + 6.7031)
    
    return salida
# muerte de vectores
def muerte_V(Tm):
    salida = 0. 
    factor = 0.0360827 #//factor a 22 grados
    salida = 8.692E-1 -(1.59E-1)*Tm + (1.116E-2)*Tm*Tm -(3.408E-4)*Tm*Tm*Tm + (3.809E-6)*Tm*Tm*Tm*Tm;
    salida = salida/factor
    
    return salida

def calculo_EV(t,Tm,V_E,G_T):
    salida = 0.
    tt = int(t)
    ##Calculo de las integrales
    #/* calculo EV */
    
    media_VE    = 1. + (0.1216*Tm*Tm - 8.66*Tm + 154.79) 
    var_VE      = 1. + (0.1728*Tm*Tm - 12.36*Tm + 230.62) 
    sigma_V     =	1./media_VE 
    k_VE        =	( media_VE*media_VE)/var_VE 
    theta_VE    =	var_VE / media_VE
    mu_V		=	muerte_V(Tm)*MU_MOSQUITA_ADULTA
    
    integral_1 = 0.
    integral_2 = 0.
    integral_3 = 0.
    integral_4 = 0.
    
    sigma_U_T  = np.zeros(tt)
    sigma_V_1  = np.zeros(tt) 
    mu_U_T     = np.zeros(tt)
    U_T        = np.zeros(tt)
    

    
    if (t > 2.):
        for j in range(0,tt):
            T_1             = Tm[j]
            media_VE_1      =	0.1216*T_1*T_1 - 8.66*T_1 + 154.79
            var_VE_1        =	0.1728*T_1*T_1 - 12.36*T_1 + 230.62
            sigma_V_1       =	1./media_VE_1
            k_VE_1          =	( media_VE_1*media_VE_1)/var_VE_1
            theta_VE_1      =	var_VE_1 / media_VE_1
            mu_V_1          =	muerte_V(T_1)*MU_MOSQUITA_ADULTA
            count_s         = 	j #importante
            sigma_U_T[j]    = 	sigma_V_1*Fbar(t, count_s , k_VE_1, theta_VE_1)*suv_exp(t, count_s, mu_V_1) 
            mu_U_T[j]       = 	mu_V_1*Fbar(t, count_s , k_VE, theta_VE)*suv_exp(t, count_s, mu_V_1)
            U_T[j]          = 	Fbar(t, count_s , k_VE, theta_VE)*suv_exp(t, count_s, mu_V_1)
            
        for j in range(1,tt):
            #print(j)
            T_1	            =	Tm[int(j-1)] 
            media_VE_1		=	0.1216*T_1*T_1 - 8.66*T_1 + 154.79
            sigma_V_1		=	1./media_VE_1 
            mu_V_1			=	muerte_V(T_1)*MU_MOSQUITA_ADULTA
            T_2				=	Tm[int(j)]
            media_VE_2		=	0.1216*T_2*T_2 - 8.66*T_2 + 154.79 
            sigma_V_2		=	1./media_VE_2 
            mu_V_2			=	muerte_V(T_2)*MU_MOSQUITA_ADULTA
            
            integral_1		=	integral_1 + 0.5*( G_T[j-1]*U_T[j-1] + G_T[j]*U_T[j] )
            integral_2		=	integral_2 + 0.5*( sigma_V_1*G_T[j-1]*U_T[j-1] + sigma_V_2*G_T[j]*U_T[j] )
            integral_3		=	integral_3 + 0.5*( G_T[j-1]*U_T[j-1] + G_T[j]*U_T[j] )
            integral_4		=	integral_4 + 0.5*( mu_V_1*G_T[j-1]*U_T[j-1] + mu_V_2*G_T[j]*U_T[j] )
    
    
    if ( integral_1 < 0.): 
        integral_1 = 0.
    if ( integral_2 < 0.):
        integral_2 = 0.
    if ( integral_3 < 0.):
        integral_3 = 0.
    if ( integral_4 < 0.):
        integral_4 = 0.
         
              
    salida 	=	sigma_V*V_E - sigma_V*integral_1 + integral_2 - mu_V*integral_3 + integral_4
    
    if ( salida < 0. ): 
        salida = 0
        
    if(Tm < NO_LATENCIA):
        salida = 0. 
    
    return salida

def runge(f,t,h,X,args=()):
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
###################################################################################

def casos(fecha,cantidad,temporada):
    days = np.zeros(len(temporada))
    ci = pd.DataFrame(days,temporada,['cantidad'])
    ci.loc[fecha] = cantidad
    return ci['cantidad']

################################ MODELO PARA LA ODE ###############################
def modelo(v,t,EV,H_t,Tmean,Tmin,Rain,CasosImp,beta_day, Kmax):
    
    dv = np.zeros(13)
    
    tt = int(t)
    
    E_D 	=	v[0]
    E_W		=	v[1]
    L		=	v[2]
    P		=	v[3]
    M		=	v[4]
    V		=	v[5]
    V_S		=	v[6] 
    V_E		=	v[7] 
    V_I		=	v[8] 
    H_S		=	v[9] 
    H_E		=	v[10] 
    H_I		=	v[11] 
    H_R		=	v[12]
    
    Tm      = Tmean[tt]
    
    rain    = Rain[tt]
    
    Tmin    = Tmin[tt]
    
    beta_day_theta_0 = beta_day*theta_T(Tm)
    
    fR      = egg_wet(rain)
    
    KL      = hogares*( Kmax*H_t/Hmax + 1.0 )
    
    m_E_C_G = 0.24*rate_mx(Tm, 10798.,100000.,14184.)*C_Gillet(L,KL)
    
    m_L     = 0.2088*rate_mx(Tm, 26018.,55990.,304.6)
    
    if (Tmin < 13.4):
        m_L = 0.
    
    mu_L    = 0.01 + 0.9725*np.exp(- (Tm - 4.85)/2.7035)
    
    C_L     = 1.5*(L/KL)
    
    m_P		=	0.384*rate_mx(Tm, 14931.,-472379.,148.)
    
    mu_P	=	0.01 + 0.9725*np.exp(- (Tm - 4.85)/2.7035)
    
    ### paramite modelo epi
    
    
    b_theta_pV	=	bite_rate*theta_T(Tm)*MIObv
		
    if ( Tm < NO_INFECCION):
        b_theta_pV = 0.
        
    mu_V    = muerte_V(Tm)*MU_MOSQUITA_ADULTA
    
    if ( Tmin < MATAR_VECTORES):
        mu_V = 2*mu_V
        
    m_M     = MADURACION_MOSQUITO
    
    mu_M    = MU_MOSQUITO_JOVEN
		
    b_theta_pH		=	bite_rate*theta_T(Tm)*MIObh 
    if ( Tmin < NO_INFECCION ): 
        b_theta_pH = 0.
        
    sigma_H			=	1./Remove_expose 
    gama			=	1./Remove_infect
    
    #deltaI = RATE_CASOS_IMP*casosImp[week]
    deltaI = RATE_CASOS_IMP*CasosImp[tt]
    
    ##modelo para la ODE
    
    dv[0]	=	beta_day_theta_0*V - fR*E_D - mu_Dry*E_D
    
    dv[1]	=	fR*E_D - m_E_C_G*E_W - mu_Wet*E_W
        
    if (Tmin < Temp_ACUATICA):
        dv[1] = MUERTE_ACUATICA*dv[1]
    
    dv[2]	=	m_E_C_G*E_W - m_L*L - ( mu_L + C_L )*L
        
    if (Tmin < Temp_ACUATICA):
        dv[1] = MUERTE_ACUATICA*dv[2]
        
    dv[3]	=	m_L*L - m_P*P - mu_P*P
        
    if (Tmin < Temp_ACUATICA):
        dv[1] = MUERTE_ACUATICA*dv[3]
    
    dv[4]	=	m_P*P - m_M*M - mu_M*M
    
    dv[5]	=	0.5*m_M*M - mu_V*V
    
    dv[6]	=	0.5*m_M*M - b_theta_pV*(H_I/poblacion)*V_S - mu_V*V_S
    
    dv[7]	=	b_theta_pV*(H_I/poblacion)*V_S - EV - mu_V*V_E
    
    dv[8]	=	EV - mu_V*V_I
    
    dv[9]	=	- b_theta_pH*(H_S/poblacion)*V_I - sigma_H*H_E
    
    dv[10]	=	b_theta_pH*(H_S/poblacion)*V_I - sigma_H*H_E
    
    dv[11]	=	sigma_H*H_E - gama*H_I + deltaI
    
    dv[12]	=	gama*H_I
    
    return dv


def fun(k, beta, data, data_rain, temporada, suma, casos_importados):

    i_temporada = temporada[0]
    f_temporada = temporada[1]
    
    i_suma = suma[0]
    f_suma = suma[1]
    
    ingreso_ci = casos_importados[0]
    cantidad_ci = casos_importados[1]

    #temporada = pd.date_range(start = i_temporada, end = f_temporada, freq='D')

    rango = data[i_temporada:f_temporada]
    rango_rain = data_rain[i_temporada:f_temporada]
    temporada = rango.index

    TMIN = rango['tmin']
    Tmean = rango['tmean']
    Rain = rango_rain['rain'] 
    HR = rango['hr']

    DAYS=np.size(Tmean)
    WEEKS = int(len(Tmean)/7) + 1

    casosIMP = casos(ingreso_ci,cantidad_ci,temporada)
    #print(casosIMP)

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

    for t in range(1,dias):
        paso_d[int(t)] = t
        h = 1.

        G_T[t]	=  bite_rate*theta_T(Tmean[t])* MIObv * v[8] * v[9]/poblacion
        G_TV[t]	=  bite_rate*theta_T(Tmean[t])* MIObv * v[6] * v[11]/poblacion
        F_T[t] = bite_rate*theta_T(Tmean[0]) * MIObh * v[8] * v[6]/poblacion

        sigma_V     =	1./(1. + (0.1216*Tmean[int(t)]*Tmean[int(t)] - 8.66*Tmean[int(t)] + 154.79) )

        EV = sigma_V*v[7]
        H_t     = hume(H_t,Rain[int(t)], Tmean[int(t)], HR[int(t)])
        
        solucion     =  runge(modelo,t,h,v,args=(EV,H_t,Tmean,TMIN,Rain,casosIMP,beta,k))
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
        
        gamma = 1./Remove_infect

        parametro[int(t)] =  np.sqrt( (sigma_V/gamma) * ((bite_rate*bite_rate*theta_T(Tmean[t])*theta_T(Tmean[t])*MIObh*MIObv)/(muerte_V(Tmean[t])*MU_MOSQUITA_ADULTA*(sigma_V + muerte_V(Tmean[t])*MU_MOSQUITA_ADULTA))) * vec_s[t] * ALPHA )

        G_T_week[week] = G_T_week[week] + G_T[t]
        #G_TV_week[week] = G_TV_week[week] + G_TV_week[t]

        contar = contar + 1
        if (contar > 6):
            G_T_week[week] = G_T_week[week]
            #G_TV_week[week] = G_TV_week[week]
            week = week + 1
            contar = 0
        
        host_i[t] = v[11]

    a = {'cantidad':G_T}
    df = pd.DataFrame(a)
    df.index = temporada[1:]
    rr = df.loc[i_suma:f_suma]
    
    return np.sum(rr)


######################### Para generar lluvias #########################
class FT:
    def __init__(self, length, alpha):
        self.length = length
        self.alpha = alpha
        self.rho = np.random.uniform()
    
    def do_1(self):
        aux = self.rho * (1-self.alpha)/self.alpha
        return self.length*aux
    
    def do_2(self):
        aux = 0.5 + ((self.rho-0.5)*(self.alpha/(1-self.alpha)))
        return self.length*aux
    
    def do_3(self):
        aux = 1 - ((1-self.rho)*((1-self.alpha)/self.alpha))
        return self.length*aux
    
    def action(self):
        if (self.rho >= 0) and (self.rho < self.alpha/2):
            return self.do_1()
        elif (
            self.rho >= self.alpha/2) and (
                self.rho <= (1-(self.alpha/2))):
            return self.do_2()
        else:
            return self.do_3()


def syntetic_rain(long, alpha, number_pieces):
    L = [long]
    N = []
    steps = math.log(number_pieces)/math.log(2)
    n = int(steps)
    for i in range(0,n):
        for j in L:
            aux = FT(j, alpha)
            a = aux.action()
            b = j-a
            N.extend([a,b])
        L = np.round(N,3)
        N = []
    if steps%1 != 0:
        while len(L) <= number_pieces - 1:
            element = random.choice(L.tolist())
            aux2 = FT(element,alpha)
            c = aux2.action()
            d = element-c
            index = np.where(L == element)[0][0]
            L[index] = c
            L = np.insert(L, index+1, d)
    return L

def anual_rain(rain_data, alpha_data):
    #diccionario con la cantidad de dias por mes
    months_days = {
        0:(31), 2:(31), 4:(31), 6:(31), 7:(31), 9:(31), 11:(31),
        1:(28),
        3:(30), 5:(30), 8:(30), 10:(30)
    }
    
    rain = np.empty([0])
    month = np.arange(0,12,1)

    for j in month:
        #la tabla de datos debe tener como primera columna la cantidad total
        #de lluvias y la segunda columna, los datos de dias de lluvia
        month_data = rain_data.iloc[j].tolist()
        total_rain = month_data[0] 
        rainy_days = int(month_data[1])
        month_days = months_days.get(j)
    
        #obtencion de los datos de alpha
        alphas = alpha_data.iloc[j].tolist()
        alpha = alphas[0]
    
        #genera la lluvia para el mes deseado
        aux = syntetic_rain(total_rain, alpha, rainy_days)
        np.random.shuffle(aux)
    
        serie_rain = np.zeros(month_days)
        aleatorio = random.sample(range(0,month_days),rainy_days)
    
        for i, valor in zip(aleatorio, aux):
            serie_rain[i] = valor
        
        rain = np.concatenate((rain,serie_rain))
    
    return rain

