# Script con condiciones iniciales

import numpy as np
from functions_new import *

# Primera condición inicial
# Constante por pedazos
def Q_0_1(x, h):
    
    # Densidad alta en x negativo y bajo en x positivo
    rho_0 = ((0.4)*(x>=1000) + 0.01*(x<1000))*rhomax
    
    # Velocidad baja en negativos y alta en positivos
    u_0 = 1*(x>=1000) + 30*(x<1000)
    
    # y inicial en funcion de u y rho
    y_0 = rho_0 * (u_0 + h(rho_0))
    
    # Vector con condición
    Q_0_ = np.zeros([2, len(y_0)])
    Q_0_[0] = rho_0
    Q_0_[1] = y_0
    
    return Q_0_


# Segunda condición inicial
# Constante por pedazos
def Q_0_2(x, h):
    
    # Densidad alta en x negativo y bajo en x positivo
    rho_0 = ((0.4)*(x<=1000) + 0.01*(x>1000))*rhomax
    
    # Velocidad baja en negativos y alta en positivos
    u_0 = 1*(x<=1000) + 30*(x>1000)
    
    # y inicial en funcion de u y rho
    y_0 = rho_0 * (u_0 + h(rho_0))
    
    # Vector con condición
    Q_0_ = np.zeros([2, len(y_0)])
    Q_0_[0] = rho_0
    Q_0_[1] = y_0
    
    return Q_0_



# Tercera condición inicial
# Gaussiana
def Q_0_3(x, h, rho_init):
    
    # Gaussiana centrada en 1500
    rho_0 = (np.exp(-((x-1000)**2)/8_000)/6+rho_init) * rhomax #np.exp(-x**2/(2*2.7**2))/(2.7*np.sqrt(2*np.pi))

    # Menor velocidad en mayor densidad
    u_0 = 1/rho_0

    # y inicial en funcion de u y rho
    y_0 = rho_0 * (u_0 + h(rho_0))

    # Vector con codicion
    Q_0_ = np.zeros([2, len(x)])
    Q_0_[0] = rho_0
    Q_0_[1] = y_0

    return Q_0_


# Cuarta condición inicial
# Constante por pedazos
def Q_0_4(x, h):

    # Densidad alta en x negativo y bajo en x positivo
    rho_0 = np.piecewise(x, [((1400 <= x) & (x <= 1500)), ((x>1500) | (x<1400))], [0.3, 0.1])*rhomax

    # Velocidad baja en negativos y alta en positivos
    u_0 = np.piecewise(x, [((1400 <= x) & (x <= 1500)), ((x>1500) | (x<1400))], [1, 19])

    # y inicial en funcion de u y rho
    y_0 = rho_0 * (u_0 + h(rho_0))

    # Vector con condición
    Q_0_ = np.zeros([2, len(y_0)])
    Q_0_[0] = rho_0
    Q_0_[1] = y_0

    return Q_0_


def Q_0_5(x, h, rho_init):
    
    # Gaussiana centrada en 1500
    rho_0 = (np.exp(-((x-100)**2)/2000)/10 +rho_init) * rhomax #np.exp(-x**2/(2*2.7**2))/(2.7*np.sqrt(2*np.pi))
    
    # Menor velocidad en mayor densidad
    u_0 = 1/rho_0
    
    # y inicial en funcion de u y rho
    y_0 = rho_0 * (u_0 + h(rho_0))
    
    # Vector con codicion
    Q_0_ = np.zeros([2, len(x)])
    Q_0_[0] = rho_0
    Q_0_[1] = y_0
    
    return Q_0_
