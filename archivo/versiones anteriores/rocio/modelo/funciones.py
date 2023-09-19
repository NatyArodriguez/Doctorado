import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # default='warn'
import scipy.special as sc
from selector_datos import seleccion, readinputdata

######################## FECHAS, INTERVALOS PARA LAS GRAFICAS ###################################
inicio = [1,'ene',2007]
final = [31,'dic',2016]
tiempo1 = np.arange('2007-01-02','2017-12-31',dtype='datetime64[D]') # para cualquier intervalo de tiempo
tiempo2 = np.arange('2009-01-01','2017-01-31',dtype='datetime64[M]') #para clinicos vs simulados (2009 al 2017)
tiempo3 = np.arange('2009-01-01','2010-01-01',dtype='datetime64[M]') #para clinicos vs simulados en un año especifico (desde 2009)
##################################################################################################



################ PARAMETROS A AJUSTAR ##################################
beta_day = 2.32     #AJUSTE ANUAL
Kmax = 190           #AJUSTE ANUAL
MIObh         = 0.75#//0.0521*2. //0.0521 //1. //
MIObv         = 0.75#//0.4867*2. //0.4867 //1. //
bite_rate = 0.27      #0.12 // 0.18 // 0.16 // 0.25 // 0.16  #0.21                                                    # 0.21
#######################################################################

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
def h_t_1(rain,Tm,Hum):
    
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
            T_1             = Tmean[j]
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
            T_1	            =	Tmean[int(j-1)] 
            media_VE_1		=	0.1216*T_1*T_1 - 8.66*T_1 + 154.79
            sigma_V_1		=	1./media_VE_1 
            mu_V_1			=	muerte_V(T_1)*MU_MOSQUITA_ADULTA
            T_2				=	Tmean[int(j)]
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
###################################################################################

################################ MODELO PARA LA ODE ###############################
def modelo(v,t, EV,H_t,beta_day, Kmax):
    
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
    
    #PARAMITES DEL CLIMA
    
    Tm      = Tmean[tt]
    
    rain    = Rain[tt]
    
    #hr      = HR[tt]
    
    Tmin    = TMIN[tt]
   
    
    ##paramite modelo eco
    
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
    deltaI = RATE_CASOS_IMP*casosImp[tt]
    
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
