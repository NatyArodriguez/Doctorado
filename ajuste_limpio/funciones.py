import numpy as np
import pandas as pd
import scipy.special as sc


###################### FUNCIONES Y PARAMETROS ######################

################ PARAMETROS A AJUSTAR ##################################
MIObh         = 0.75#//0.0521*2. //0.0521 //1. //
MIObv         = 0.75#//0.4867*2. //0.4867 //1. //
bite_rate = 0.27      #0.12 // 0.18 // 0.16 // 0.25 // 0.16  #0.21                                                    # 0.21
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

casos_confirmados = np.concatenate((casos_2009,casos_2010,casos_2011,casos_2012,casos_2013,casos_2014,casos_2015,casos_2016))


d = {'casos_2009': casos_2009,
     'casos_2010': casos_2010,
     'casos_2011': casos_2011,
     'casos_2012': casos_2012,
     'casos_2013': casos_2013,
     'casos_2014': casos_2014,
     'casos_2015': casos_2015,
     'casos_2016': casos_2016}

df = pd.DataFrame(d)
#############################################################

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
def casos_temporada(v,año):
    ME = np.zeros(12)
    if año == 2009:
        ME[0] = sum(v[81:85]) #JUL08
        ME[1] = sum(v[85:89]) #AGO08
        ME[2] = sum(v[89:93]) #SEP08
        ME[3] = sum(v[93:97]) #OCT08
        ME[4] = sum(v[97:101]) #NOV08
        ME[5] = sum(v[101:105]) #DIC08
        ME[6] = sum(v[105:109])#ENE09
        ME[7] = sum(v[109:113])#FEB09
        ME[8] = sum(v[113:118])#MAR09
        ME[9] = sum(v[118:122])#ABR09
        ME[10] = sum(v[122:127])#MAY09
        ME[11] = sum(v[127:131])#JUN09
    
    if año == 2010:
        ME[0] = sum(v[131:135])#JUL09
        ME[1] = sum(v[135:140])#AGO09
        ME[2] = sum(v[140:144])#SEP09
        ME[3] = sum(v[144:149])#OCT09
        ME[4] = sum(v[149:153])#NOV09
        ME[5] = sum(v[153:157])#DIC09
        ME[6] = sum(v[157:161])#ENE10
        ME[7] = sum(v[161:165])#FEB10
        ME[8] = sum(v[165:170])#MAR10
        ME[9] = sum(v[170:174])#ABR10
        ME[10] = sum(v[174:178])#MAY10
        ME[11] = sum(v[178:183])#JUN10

    if año == 2011:

        ME[0] = sum(v[183:187])#JUL10
        ME[1] = sum(v[187:192])#AGO10
        ME[2] = sum(v[192:196])#SEP10
        ME[3] = sum(v[196:200])#OCT10
        ME[4] = sum(v[200:204])#NOV10
        ME[5] = sum(v[204:209])#DIC10
        ME[6] = sum(v[209:214])#ENE11
        ME[7] = sum(v[214:218])#FEB11
        ME[8] = sum(v[218:222])#MAR11
        ME[9] = sum(v[222:226])#ABR11
        ME[10] = sum(v[226:231])#MAY11
        ME[11] = sum(v[231:235])#JUN11
    
    if año == 2012:

        ME[0] = sum(v[235:240])#JUL11
        ME[1] = sum(v[240:244])#AGO11
        ME[2] = sum(v[244:248])#SEP11
        ME[3] = sum(v[248:253])#OCT11
        ME[4] = sum(v[253:257])#NOV11
        ME[5] = sum(v[257:261])#DIC11
        ME[6] = sum(v[261:266])#ENE12
        ME[7] = sum(v[266:270])#FEB12
        ME[8] = sum(v[270:274])#MAR12
        ME[9] = sum(v[274:279])#ABR12
        ME[10] = sum(v[279:283])#MAY12
        ME[11] = sum(v[283:287])#JUN12
    
    if año == 2013:

        ME[0] = sum(v[287:292])#JUL12
        ME[1] = sum(v[292:296])#AGO12
        ME[2] = sum(v[296:301])#SEP12
        ME[3] = sum(v[301:305])#OCT12
        ME[4] = sum(v[305:309])#NOV12
        ME[5] = sum(v[309:313])#DIC12
        ME[6] = sum(v[313:318])#ENE13
        ME[7] = sum(v[318:322])#FEB13
        ME[8] = sum(v[322:327])#MAR13
        ME[9] = sum(v[327:331])#ABR13
        ME[10] = sum(v[331:335])#MAY13
        ME[11] = sum(v[335:340])#JUN13
    
    if año == 2014:

        ME[0] = sum(v[340:344])#JUL13
        ME[1] = sum(v[344:348])#AGO13
        ME[2] = sum(v[348:353])#SEP13
        ME[3] = sum(v[353:357])#OCT13
        ME[4] = sum(v[357:361])#NOV13
        ME[5] = sum(v[361:365])#DIC13
        ME[6] = sum(v[365:370])#ENE14
        ME[7] = sum(v[370:374])#FEB14
        ME[8] = sum(v[374:379])#MAR14
        ME[9] = sum(v[379:383])#ABR14
        ME[10] = sum(v[383:387])#MAY14
        ME[11] = sum(v[387:392])#JUN14
    
    if año == 2015:

        ME[0] = sum(v[392:396])#JUL14
        ME[1] = sum(v[396:401])#AGO14
        ME[2] = sum(v[401:405])#SEP14
        ME[3] = sum(v[405:409])#OCT14
        ME[4] = sum(v[409:414])#NOV14
        ME[5] = sum(v[414:418])#DIC14
        ME[6] = sum(v[418:422])#ENE15
        ME[7] = sum(v[422:426])#FEB15
        ME[8] = sum(v[426:431])#MAR15
        ME[9] = sum(v[431:435])#ABR15
        ME[10] = sum(v[435:440])#MAY15
        ME[11] = sum(v[440:444])#JUN15

    
    if año == 2016:
        ME[0] = sum(v[444:448])#JUL15
        ME[1] = sum(v[448:453])#AGO15
        ME[2] = sum(v[453:457])#SEP15
        ME[3] = sum(v[457:461])#OCT15
        ME[4] = sum(v[461:466])#NOV15
        ME[5] = sum(v[466:470])#DIC15
        ME[6] = sum(v[470:475])#ENE16
        ME[7] = sum(v[475:479])#FEB16
        ME[8] = sum(v[479:483])#MAR16
        ME[9] = sum(v[483:487])#ABR16
        ME[10] = sum(v[487:492])#MAY16
        ME[11] = sum(v[492:496])#JUN16
    return (ME)
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