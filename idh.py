from re import L
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
import matplotlib.dates as mdates
from semanas_epi import orden
from semanas_epi import orden_years
from semanas_epi import orden_years_medio
from semanas_epi import elegir
from datetime import datetime, timedelta
from matplotlib.ticker import NullFormatter
from matplotlib.dates import MonthLocator, DateFormatter


################## PARA LEER LOS ARCHIVOS #######################
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
#################################################################

hogares=17633
lee=readinputdata('huevos_secos_2001.txt')
HS=lee[:,0]
data=readinputdata("Oran_2001_2017.txt")
Tmin=data[:,0]
Tmax=data[:,1]
Tmean=data[:,2]
Tmedian=data[:,3]
Rain=data[:,4]
Hum_mean=data[:,5]
Hum_median=data[:,6]

largo=len(Tmedian)

año=2017
IDH=orden_years_medio(HS,año)/hogares
np.savetxt("idh_anual.txt",IDH,fmt='%1.4e')
a=len(IDH)
LLUVIA=orden_years_medio(Rain, año)
#print(a)
x=np.linspace(0,a,a)
xx=np.linspace(0,largo,largo)




################### GRAFICA IDH ##########################
"""
plt.plot(xx,Rain,color='m',label='Rain')
plt.legend()
plt.grid(True)
plt.show()

dates=list()
dates.append(datetime.strptime('2012-01-01','%Y-%m-%d'))
for d in x[1:]:
    dates.append(dates[0]+timedelta(days=d))

plt.plot(dates,IDH,label='IDH')
plt.plot(dates,LLUVIA,label='LLUVIAS')
#plt.legend()
ax=plt.gca()
ax.legend()
ax.grid(True)
ax.set_xlim([dates[0],dates[-1]])
ax.xaxis.set_major_locator(MonthLocator())
ax.xaxis.set_minor_locator(MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(NullFormatter())
ax.xaxis.set_minor_formatter(DateFormatter('%b'))
plt.show()
"""
##########################################################

############## COMPARACION CON IDH MEDIO PROMEDIO ##################################

datos= pd.read_table("IDH_COMPARACION.txt",sep='\t')


año = datos.a2006
prom    = datos.PROM
tt=len(año)
t=np.linspace(0,tt,tt)

"""
plt.plot(t,prom,color='b',label='PROMEDIO MEDIO')
plt.plot(t,año,color='m',label='2016')
plt.legend()
plt.grid(True)
plt.show()
"""

dates=list()
dates.append(datetime.strptime('2015-01-01','%Y-%m-%d'))
for d in x[1:]:
    dates.append(dates[0]+timedelta(days=d))

plt.plot(dates,prom,label='Promedio Medio')
plt.plot(dates,año,label='2015')
#plt.legend()
ax=plt.gca()
ax.legend()
ax.grid(True)
ax.set_xlim([dates[0],dates[-1]])
ax.xaxis.set_major_locator(MonthLocator())
ax.xaxis.set_minor_locator(MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(NullFormatter())
ax.xaxis.set_minor_formatter(DateFormatter('%b'))
plt.show()

#####################################################################################

####################### PRUEBA CON EL PAPER ####################################
"""
IDH_2006_2007=elegir(HS,2006,2007)/hogares
p1=9.05
p2=4.17
p3=8.87
p4=4.40
p5=5.47
p6=3.22
p7=7.63
l=len(IDH_2006_2007)
#t=np.linspace(0,l,l)

t = mdates.drange(mdates.datetime.date(2005,12,31), mdates.datetime.date(2007,12,31), delta=mdates.datetime.timedelta(days=1))
#print (t)
fig,ax=plt.subplots()
ax.plot(t,IDH_2006_2007)
ax.plot(t[312],p1,marker='o',color='black')
ax.plot(t[326],p2,marker='o',color='black')
ax.plot(t[340],p3,marker='o',color='black')
ax.plot(t[367],p4,marker='o',color='black')
ax.plot(t[381],p5,marker='o',color='black')
ax.plot(t[396],p6,marker='o',color='black')
ax.plot(t[423],p7,marker='o',color='black')

ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.show()
"""