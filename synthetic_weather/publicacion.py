import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import funciones_actualizado as fn
import time

#tabla con datos de la lluvia tipica
tipic_rain = pd.read_csv('tipic_rain.txt', sep='\t')

#tabla de datos de alpha historico
tipic_alpha = fn.hystoric_al

#De oran_medio voy a usar la tmean, hum, tmin
oran_medio = fn.df_medio #-->2001-2017

#me voy a solucionar la vida y volaron los años bisiestos
df_oran = oran_medio[~((oran_medio.index.month == 2) & (oran_medio.index.day == 29))]
indices = df_oran.index

temporada = ['2001-01-01', '2008-12-31'] #temporada total de datos
suma_casos = ['2007-07-01', '2008-06-30'] #en que periodo de tiempo deseamos obtener la cantidad total de nuevos casos
ci_sim = np.arange('2007-07-01','2008-06-30',dtype='datetime64[D]') #rango de ingreso de los casos importados

#tamaño de una matriz para guardar los resultados
n_sim  = ci_sim.size
iteraciones = 100
matriz = np.zeros([iteraciones,n_sim])

'''
start_time = time.time()
for i in range(n_sim):
    ci_info = [ci_sim[i],1]
    for j in range(iteraciones):
        rain_serie = fn.anual_rain(tipic_rain,tipic_alpha)
        rr = np.tile(rain_serie,17)
        rain = {'rain':rr}
        df_rain = pd.DataFrame(rain)
        df_rain.index = indices
        aux = fn.fun(350, 1.69, df_oran, df_rain, temporada, suma_casos, ci_info)
        matriz[j,i] = aux

#para guardar los resultados en un archivo
with open('final_size.txt', 'w') as final_size:
    final_size.write('\t'.join([str(date) for date in ci_sim]) + '\n')
    for i in range(iteraciones):
        final_size.write('\t'.join([f'{matriz[i,j]:.2f}' for j in range(n_sim)]) + '\n')

end_time = time.time()
print(f'El codigo tomo {end_time - start_time} segundos.')

# 10 iteraciones --> 794.83s ~ 13min
# 100 iteraciones --> 8007.63s ~ 2,22h
'''

resultados = pd.read_csv('final_size.txt', sep='\t', header=0)
mean = resultados.mean()

df = pd.read_table('fsize_his.txt', sep='\t', names=['250', '300', '350', '400',
                                                     '450', '500', 'nan'])

mean.plot(label='with synthetic rain, k=350')
df['350'].plot(label='k=350')
df['300'].plot(label='k=300')
plt.title('Epidemic Size')
plt.ylabel('Size')
plt.legend()
plt.show()
