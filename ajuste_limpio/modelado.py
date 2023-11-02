import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dias = np.arange('2001-01-01','2018-01-01',dtype='datetime64[D]')
temporada = np.arange('2008-07-01','2016-07-01',dtype='datetime64[D]')
temp9 = np.arange('2008-07-01','2009-07-01',dtype='datetime64[D]')
temp10 = np.arange('2009-07-01','2010-07-01',dtype='datetime64[D]')
temp11 = np.arange('2010-07-01','2011-07-01',dtype='datetime64[D]')
temp12 = np.arange('2011-07-01','2012-07-01',dtype='datetime64[D]') #366
temp13 = np.arange('2012-07-01','2013-07-01',dtype='datetime64[D]')
temp14 = np.arange('2013-07-01','2014-07-01',dtype='datetime64[D]')
temp15 = np.arange('2014-07-01','2015-07-01',dtype='datetime64[D]')
temp16 = np.arange('2015-07-01','2016-07-01',dtype='datetime64[D]') #366

##################################################################
b7 = b8 = b9 = 2.40#2.73#2.17#2.55#2.07#2.71#1.47#2.799#ver[1]#1.86#1.447 #1.47
b10 = 1.69#1.40#1.26#1.73#1.86#1.08#1.39#1.00#1.1664#ver[2]#0.77 #2.799 #1.018
b11 = 1.54#1.59#1.38#1.74#1.50#1.55#1.499#1.944#ver[3]#0.05 #1.732 #2.72
b12 = 1.80#2.66#1.90#1.56#1.58#1.94#1.300#2.104#1.2#ver[4]#2.318 #2.175#1.498
b13 = 1.68#1.36#1.50#1.73#1.93#2.56#1.094#2.067#ver[5]#2.157 #1.622#1.96
b14 = 2.1#2.62#2.00#2.15#2.61#1.98#1.313#1.507#ver[6]#1.127 #2.2339#1.39 #1.099
b15 = 1.65#1.92#1.50#1.64#1.67#1.76#1.70#1.799#1.804#ver[7]#0.018 #1.5868#2.15
b16 = 2.52#2.00#2.80#2.64#2.80#2.67#2.656#2.799#ver[8]#0.56 #2.721#1.74
b17 = b16

k7 = k8 = k9 = 306.03#236.80#385.03#271.52#434.86#239.95#231.52 #227.559#ver[9]#96.60 #2678.696#231.52
k10 = 449.96#356.44#393.49#59.73#344.81#864.65#836.61#981.309#ver[10]#1147.85 #364.395#567.89
k11 = 500.89#177.71#378.36#250.96#444.83#114.21#499.983#612.393#ver[11]#53.43 #1206.935#1181.72
k12 = 800.02#241.89#453.58#724.13#302.54#74.11#600.014#650.037#ver[12]#260.63 #1924.660#17.915
k13 = 819.99#437.23#751.75#699.48#779.29#255.64#998.261#999.577#300#ver[13]#118.908 #1360.4268#1981.005
k14 = 700.00#550.62#676.53#724.67#406.39#742.37#999.906#999.574#ver[14]#2886.399 #2723.57788#183.61 #181.7356
k15 = 359.99#459.25#467.11#490.92#370.90#475.68#214.93#599.99#338.839#ver[15]#95.606 #2849.243#1681.09
k16 = 600.06#848.55#494.10#526.81#447.82#550.85#63.755#504.040#ver[16]#1437.978 #1531.4172#52.21
k17 = k16
k = [k9,k10,k11,k12,k13,k14,k15,k16]
beta_day = [b9,b10,b11,b12,b13,b14,b15,b16]
###############################################################################

days = np.size(temporada)
oran = pd.read_table('ORAN.txt',sep='\t')
oran.index = dias
no , mo = oran.shape #(6209,7)

oran_medio = pd.read_table('Oran_2001_2017_medio.txt',sep='\t')
oran_medio.index = dias

rango = oran.loc[temporada]
lluvia = rango['rain']

dias_rain = []
for i in lluvia:
    if i > 0:
        dias_rain.append(1)
    else:
        dias_rain.append(0)


#prueba = oran.loc[temp9]
#filt =prueba[ prueba['rain'] > 0.]['rain']
#print(filt)

rain09 = oran.loc[temp9]
rain10 = oran.loc[temp10]
rain11 = oran.loc[temp11]
rain12 = oran.loc[temp12]
rain13 = oran.loc[temp13]
rain14 = oran.loc[temp14]
rain15 = oran.loc[temp15]
rain16 = oran.loc[temp16]

maximos = [max(rain09['rain']),max(rain10['rain']),max(rain11['rain']),max(rain12['rain']),max(rain13['rain']),max(rain14['rain']),max(rain15['rain']),max(rain16['rain'])]
minimos = [min(rain09['rain']),min(rain10['rain']),min(rain11['rain']),min(rain12['rain']),min(rain13['rain']),min(rain14['rain']),min(rain15['rain']),min(rain16['rain'])]
up = np.mean(maximos)
down = np.mean(minimos)
low = 0.5

days_rain = np.array([(rain09[(rain09['rain']>low) & (rain09['rain']<up)])['rain'].mean(),
                     (rain10[(rain10['rain']>low) & (rain10['rain']<up)])['rain'].mean(),
                     #(rain11[(rain11['rain']>low) & (rain11['rain']<up)])['rain'].mean(),
                     (rain12[(rain12['rain']>low) & (rain12['rain']<up)])['rain'].mean(),
                     (rain13[(rain13['rain']>low) & (rain13['rain']<up)])['rain'].mean(),
                     (rain14[(rain14['rain']>low) & (rain14['rain']<up)])['rain'].mean(),
                     #(rain15[(rain15['rain']>low) & (rain15['rain']<up)])['rain'].mean(),
                     (rain16[(rain16['rain']>low) & (rain16['rain']<up)])['rain'].mean()])

rain_mean = np.array([(oran.loc[temp9])['rain'].mean(),
             (oran.loc[temp10])['rain'].mean(),
             (oran.loc[temp11])['rain'].mean(),
             (oran.loc[temp12])['rain'].mean(),
             (oran.loc[temp13])['rain'].mean(),
             (oran.loc[temp14])['rain'].mean(),
             (oran.loc[temp15])['rain'].mean(),
             (oran.loc[temp16])['rain'].mean()])


rain_mean_hist = np.array([(oran_medio.loc[temp9])['rain'].mean(),
             (oran_medio.loc[temp10])['rain'].mean(),
             (oran_medio.loc[temp11])['rain'].mean(),
             (oran_medio.loc[temp12])['rain'].mean(),
             (oran_medio.loc[temp13])['rain'].mean(),
             (oran_medio.loc[temp14])['rain'].mean(),
             (oran_medio.loc[temp15])['rain'].mean(),
             (oran_medio.loc[temp16])['rain'].mean()])
#print(days_rain)
#print(rain_mean_hist)

funcion = 1/(np.exp(rain_mean))
funcion1 = 1/(np.log(days_rain))
dif = (rain_mean - rain_mean_hist)

def f(x):
    #y = 800*(1-np.exp(-x))/(1+np.exp(-x))
    a=np.array(x)
    y = np.empty_like(a)
    for i in range(0,8):
        y[int(i)] = 1/(np.exp((a[i]-rain_mean_hist[0])/(1/1000*rain_mean_hist[0]))+1)
    return (y)

#poly = np.polyfit(np.exp(dif),beta_day,4)
#curv = np.poly1d(poly)
#rango = np.linspace(0.2,3.5,100)

kk = [k9,k10,k11,k12,k13,k14,k15,k16]

#print(rain_mean)
#print(funcion)
#plt.scatter(funcion1,beta_day,label = 'k vs rain')
#plt.scatter((1/(np.exp(days_rain))*100),k)

#plt.scatter(funcion,beta_day)
#plt.plot(rango,curv(rango))

#print(rain_mean)
#plt.scatter(k,beta_day)
#plt.scatter(1/np.exp(dif),k)
#plt.scatter(dif,k,marker='<')
#plt.plot(temporada,dias_rain)
#plt.scatter(rain_mean,k)
plt.scatter(rain_mean,f(rain_mean))
#plt.title()
#plt.xlabel('exp(rain hist - rain)')
#plt.ylabel('beta_day')
plt.legend()
plt.show()

