import numpy as np
import matplotlib.pyplot as plt

def funcion(x):
	return np.exp(-x) - x
def dfuncion(x):
	return - np.exp(-x) - 1.

def Biseccion(a,b,err,funcion):
	c = 0.5*(a + b)
	while np.abs(funcion(c)) > err:
		if funcion(a)*funcion(c) < 0.:
			b = c
		else:
			a = c
		c = 0.5*(a + b)
	return c
def Newton(seed,err,funcion,dfuncion):
	c = seed
	while np.abs(funcion(c)) > err:
		c = c - funcion(c)/dfuncion(c)
	return c
def Secante(a,b,err,funcion):
	c = 0.5*(a +b)
	while np.abs(funcion(c)) > err:
		c  = b - funcion(b)*(b - a)/(funcion(b) - funcion(a))
		a = b
		b = c
	return c
def __main__00__(a,b):
	err = 1.E-9
	r = Biseccion(a,b,err,funcion)
	print("el valor de la raiz por biseccion es %.9f"%r)
	r = Newton(0.5,err,funcion,dfuncion)
	print("el valor de la raiz por newton es %.9f"%r)
	r = Secante(a,b,err,funcion)
	print("el valor de la raiz por secante es %.9f"%r)
	return True

x = np.linspace(0.,2.,100)
plt.plot(x,funcion(x))
plt.show()
__main__00__(0.,2.)

