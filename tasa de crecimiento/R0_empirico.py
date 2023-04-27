import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # default='warn'
import scipy.special as sc
from selector_datos import seleccion
import matplotlib.pyplot as plt

from modelo2 import G_T, tiempo1, vector

def divisiones(x):
    posicion = np.where(x!=0.)
    coleccion=[]
    coleccion.append(x[0]) #agrego el primer elemento para que todos los vectores tengan el mismo tamaño
    for i in range(1,np.size(x)):
        if i in posicion[0]:
            if (i-1) in posicion[0]:
                dividendo = x[i]
                divisor = x[i-1]
                elemento = dividendo/divisor
                coleccion.append(elemento)
            else:
                coleccion.append(0.)
        else:
            coleccion.append(0.)
    return(coleccion)

#a = np.array([1,12,0,31,0,0,3,5])
#veamos = divisiones(a)
tasa = divisiones(G_T)


plt.figure(figsize=(10,6))
plt.plot(tiempo1,tasa,'r', label='Tasa de crecimiento')
plt.plot(tiempo1, G_T, '--b', label='Nuevos casos')
plt.grid()
plt.legend()
plt.title(vector)
plt.show()



################ Primera prueba ##########################
""" 
R0_empirico=[]
for i in range(0,np.size(G_T)-1):
    if G_T[i] == 0:
        R0_empirico.append(0.)
    if G_T[i] > 0. :
        divisor = G_T[i]
        dividendo = G_T[i + 1]
        R0_empirico.append(dividendo/divisor)

#plt.plot(tiempo1[1:],R0_empirico,tiempo1[1:],G_T[1:])
"""