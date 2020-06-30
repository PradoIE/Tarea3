# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:33:47 2020

@author: alons
"""


import matplotlib
import numpy as np
from pylab import *
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import*
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d.axes3d import Axes3D



datos = pd.read_csv('xy.csv',header=0)
datosx= pd.read_csv('yx.csv',header=0) # SE TRASPORTAN LOS VALORES DE X A Y Y VICEVERZA EN EXCELL PARA LEERLOS DE UNA MEJOR MANERA
datosp = pd.read_csv('xyp.csv',header=1)
print('############################################################################ Punto 1')

#############################################################################################
# SUMAS DE LAS COLUMNAS Y

Y5  = datos['y5'].sum()
Y6  = datos['y6'].sum()
Y7  = datos['y7'].sum()
Y8  = datos['y8'].sum()
Y9  = datos['y9'].sum()
Y10 = datos['y10'].sum()
Y11 = datos['y11'].sum()
Y12 = datos['y12'].sum()
Y13 = datos['y13'].sum()
Y14 = datos['y14'].sum()
Y15 = datos['y15'].sum()
Y16 = datos['y16'].sum()
Y17 = datos['y17'].sum()
Y18 = datos['y18'].sum()
Y19 = datos['y19'].sum()
Y20 = datos['y20'].sum()
Y21 = datos['y21'].sum()
Y22 = datos['y22'].sum()
Y23 = datos['y23'].sum()
Y24 = datos['y24'].sum()
Y25 = datos['y25'].sum()

#############################################################################################
# SUMAS DE LAS FILAS X

X5  = datosx['x5'].sum()
X6  = datosx['x6'].sum()
X7  = datosx['x7'].sum()
X8  = datosx['x8'].sum()
X9  = datosx['x9'].sum()
X10 = datosx['x10'].sum()
X11 = datosx['x11'].sum()
X12 = datosx['x12'].sum()
X13 = datosx['x13'].sum()
X14 = datosx['x14'].sum()
X15 = datosx['x15'].sum()

#############################################################################################
# SE CRAN LOS ARRAYS Y SE GRÁFICAN

Y = [Y5,Y6,Y7,Y8,Y9,Y10,Y11,Y12,Y13,Y14,Y15,Y16,Y17,Y18,Y19,Y20,Y21,Y22,Y23,Y24,Y25]
X = [X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15]

ys = np.linspace(5,25,21) # SÉRIE DE DATOS PARA LA GRÁFICA DE Y
xs = np.linspace(5,15,11) # SÉRIE DE DATOS PARA LA GRÁFICA DE X

#GRÁFICAS
plt.plot(xs,X)
plt.plot(ys,Y)

#XS  = datos.sum(axis=1)
#YS  = datos.sum(axis=0)

#print(XS)
#print(YS)

#############################################################################################
# SE BUSCAN LOS PARÁRAMETROS DE LA FUNCIÓN DE DENSIDAD DE MEJOR AJUSTE PARA LAS FUNCIONES DE X Y Y
# En este caso ya que las gráficas son simétricas en su eje, se puede decir que la curva de mejor ajuste es la de Gauss 

def gaussiana(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2/(2*sigma**2))

paramX, _ = curve_fit(gaussiana,xs,X)
paramY, _ = curve_fit(gaussiana,ys,Y)
print('Mu y sigma para X',paramX)
print('Mu y sigma para Y',paramY)
print('')
print('############################################################################ Punto2')

# CÓDIGO DE WOLFRAM PARA COMPARAR LAS GRÁFICAS
# graph ((1)/((6.02693813)*(2*pi)^(1/2)))*e^((-(x-10.07946091)^(2))/(2*(6.02693813)^2))



#############################################################################################
# FUNCIÓN DE DENSIDAD CONJUNTA WOLRAM
print('0.00800355 e^(-0.0459292 (-9.90484 + x)^2 - 0.013765 (-15.0795 + y)^2)')

print('')

print('############################################################################ Punto 3')


# FUNCIÓN DE DENSIDAD CONJUNTA PYTHON
# 0.00800355 np.exp(-0.0459292 (-9.90484 + x)**2 - 0.013765 (-15.0795 + y)**2)

#############################################################################################

############################### CORRELACIÓN #############################

#DEFINO LA MULTIPLICACIÓN ENTRE X, Y y P

datosp=pd.read_csv("xyp.csv") #LEER EL NUEVO ARCHIVO

datosp['productoFILA'] = datosp['x']*datosp['y']*datosp['p'] # MULTIPLICACION DE CADA FILA

correlacion  = datosp['productoFILA'].sum() # SUMA DE LAS MULTIPLICACIONES

print('El valor de la correlación es de = \n',correlacion )

print('Que este valor de alto significa que las variables aleatorias se encuentran linealmente asociadas.')

print('')
print('##############################')



############################### COOVARIANZA #############################

# LA MEDIA ES EL VALOR DE MU DE CADA UNA

MEDIAX = 9.90484381
MEDIAY = 15.0794609

datosp['productoFILA2'] = (datosp['x']-MEDIAX)*(datosp['y']-MEDIAY)*datosp['p'] # MULTIPLICACION DE CADA FILA

coovarianza  = datosp['productoFILA2'].sum() # SUMA DE LAS MULTIPLICACIONES

print('El valor de la coovarianza es de = \n',coovarianza)
print('Dado que el valor es positivo significa que las varables presentan una relación directa, y se puede decir que si una tiene un valor alto, la otra también.')
print('')
print('##############################')


############################### PEARSON #############################

SIGMAX = 3.29944288
SIGMAY = 6.02693774

datosp['E'] = (datosp['x']-MEDIAX)*(datosp['y']-MEDIAY)*datosp['p'] # VALOR ESPERADO - LA MEDIA

EM = datosp['E'].sum()

pearson = EM/(SIGMAX*SIGMAY)

print('El valor de la correlación de pearson es de = \n' ,pearson)

print('Este dato lo que confirma es una cierta dependecia de las variables una de la otra')

print('')

print('############################################################################ Punto 4')
#############################################################################################
# SE HACEN LAS GRÁFICAS EN 2D Y 3D 

X1 = np.linspace(0,20,65)
Y1 = np.linspace(0,30,65)
X  = np.linspace(0,20,65)
Y  = np.linspace(0,30,65)

X,Y = np.meshgrid(X,Y)
X1,Y1 = np.meshgrid(X1,Y1)

Z1 = 0.5

Z = (0.00800355)*np.exp(((-0.0459292)*(-9.90484 + X)**2) - ((0.013765)*(-15.0795 + Y)**2))

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_wireframe(X,Y,Z,    rstride=2, cstride=2, cmap = 'Blues')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('FX')

show()