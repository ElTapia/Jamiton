# Script con condiciones iniciales

import numpy as np
from functions_new import *
from jamiton_gen import *

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


# sexta condición inicial
# Jamiton teórico

def Q_0_jam(h, N, tau, rho_s, plotear=False):

    x_minus, x_plus, sol_rho, sol_u, sol_rho_eta, sol_u_eta, s = init_program(tau, rho_s, plotear)
    L = np.fabs(x_minus-x_plus)

    print("x_+: ", x_plus)
    print("x_-: ", x_minus)
    print("Largo del intervalo: ", L)
    dx = L/N  # float(input("Ingrese dx: "))

    # N = int((x_minus-x_plus)//dx)
    # x_to_solve = np.linspace(x_plus, x_minus, N)
    x_to_plot = np.arange(0, L, dx)

    def rho_per(x):
        interval = x_minus - x_plus
        x_per = (x - x_plus) % interval + x_plus
        return sol_rho(x_per)

    def u_per(x):
        interval = x_minus - x_plus
        x_per = (x - x_minus) % interval + x_plus
        return sol_u(x_per)
    
    rho_0 = rho_per(x_to_plot)
    u_0 = u_per(x_to_plot)
    y_0 = rho_0 * (u_0 + h(rho_0))

    Q_0_ = np.zeros([2, N])
    Q_0_[0] = rho_0
    Q_0_[1] = y_0

    def teo_rho(x, t, s=s):
        eta_minus = x_minus / tau
        eta_plus = x_plus / tau
        eta = (x - s*t)/tau
        interval = eta_minus - eta_plus
        eta_per = (eta - eta_plus) % interval + eta_plus

        return sol_rho_eta(eta_per)

    def teo_u(x, t, s=s):
        eta_minus = x_minus / tau
        eta_plus = x_plus / tau
        eta = (x - s*t)/tau
        interval = eta_minus - eta_plus
        eta_per = (eta - eta_plus) % interval + eta_plus

        return sol_u_eta(eta_per)

    return Q_0_, x_to_plot, teo_rho, teo_u



def Q_0_collide(h, N, tau, rho_s_1, rho_s_2, plotear=False, x_init=None):
    # Primer jamiton
    x_minus_1, x_plus_1, sol_rho_1, sol_u_1, sol_rho_eta_1, sol_u_eta_1, s_1 = init_program(tau, rho_s_1, plotear, x_init)

    # Segundo jamiton
    x_minus_2, x_plus_2, sol_rho_2, sol_u_2, sol_rho_eta_2, sol_u_eta_2, s_2 = init_program(tau, rho_s_2, plotear, x_init)

    # Jamitones compatibles
    rho_min_1 = sol_rho_1(x_minus_1)
    rho_min_2 = sol_rho_2(x_minus_2)

    compatible = np.isclose(rho_min_1, rho_min_2)

    if not compatible:
        print("Jamitones incompatibles")
        return None

    L = x_minus_2 + np.fabs(x_minus_1 - x_plus_2) - x_plus_1
    print("x_+: ", x_plus_1)
    print("x_-: ", x_minus_2)
    print("Largo del intervalo: ", L)  # np.fabs(x_minus_2 - x_plus_1))
    dx = L/N

    # Intervalo de solución
    # N = int((x_minus_2+(x_minus_1 - x_plus_2) - x_plus_1)//dx)
    #x_to_plot = np.arange(x_plus_1, x_minus_2+(x_minus_1 - x_plus_2), dx)
    x_to_plot = np.arange(0, L, dx) # x_minus_2 + np.fabs(x_minus_1 - x_plus_2) - x_plus_1

    def rho_sol_combined(x):
        if x_plus_1 <= x and x <= x_minus_1:
            return sol_rho_1(x)
    
        elif x_plus_2 <= x - np.fabs(x_minus_1 - x_plus_2) and x - np.fabs(x_minus_1 - x_plus_2) <= x_minus_2:
            return sol_rho_2(x-(x_minus_1 - x_plus_2))

    def u_sol_combined(x):
        if x_plus_1 <= x and x <= x_minus_1:
            return sol_u_1(x)
    
        elif x_plus_2 <= x - np.fabs(x_minus_1 - x_plus_2) and x - np.fabs(x_minus_1 - x_plus_2) <= x_minus_2:
            return sol_u_2(x-(x_minus_1 - x_plus_2))

    rho_sol_combined = np.vectorize(rho_sol_combined)
    u_sol_combined = np.vectorize(u_sol_combined)

    def rho_per(x):
        interval = x_minus_2 + np.fabs(x_minus_1 - x_plus_2) - x_plus_1
        x_per = (x - x_plus_1) % interval + x_plus_1
        return rho_sol_combined(x_per)

    def u_per(x):
        interval = x_minus_2 + np.fabs(x_minus_1 - x_plus_2) - x_plus_1
        x_per = (x - x_plus_1) % interval + x_plus_1
        return u_sol_combined(x_per)

    rho_0 = rho_per(x_to_plot)
    u_0 = u_per(x_to_plot)
    y_0 = rho_0 * (u_0 + h(rho_0))

    Q_0_ = np.zeros([2, len(x_to_plot)])
    Q_0_[0] = rho_0
    Q_0_[1] = y_0

    return Q_0_, x_to_plot



def Q_0_collide_triple(h, N, tau, rho_s_1, rho_s_2, rho_s_3, plotear=False, x_init=None):
    # Primer jamiton
    x_minus_1, x_plus_1, sol_rho_1, sol_u_1, sol_rho_eta_1, sol_u_eta_1, s_1 = init_program(tau, rho_s_1, plotear, x_init)

    # Segundo jamiton
    x_minus_2, x_plus_2, sol_rho_2, sol_u_2, sol_rho_eta_2, sol_u_eta_2, s_2 = init_program(tau, rho_s_2, plotear, x_init)

    # Tercer jamiton
    x_minus_3, x_plus_3, sol_rho_3, sol_u_3, sol_rho_eta_3, sol_u_eta_3, s_3 = init_program(tau, rho_s_3, plotear, x_init)

    # Jamitones compatibles
    rho_min_1 = sol_rho_1(x_minus_1)
    rho_min_2 = sol_rho_2(x_minus_2)
    rho_min_3 = sol_rho_2(x_minus_3)

    compatible_1 = np.isclose(rho_min_1, rho_min_2)
    compatible_2 = np.isclose(rho_min_2, rho_min_3)
    compatible_3 = np.isclose(rho_min_1, rho_min_3)

    if not compatible_1 or not compatible_2 or not compatible_3:
        print("Jamitones incompatibles")
        return None

    x_minus_max = np.max(x_minus_1, x_minus_2, x_minus_3)
    x_plus_min = np.min(x_plus_1, x_plus_2, x_plus_3)

    L = x_minus_max - x_plus_min
    #print("x_+: ", x_plus_1)
    #print("x_-: ", x_minus_2)
    print("Largo del intervalo: ", L)
    dx = L/N

    # Intervalo de solución
    # N = int((x_minus_2+(x_minus_1 - x_plus_2) - x_plus_1)//dx)
    # x_to_solve = np.linspace(x_plus_1, x_minus_2+(x_minus_1 - x_plus_2), N)

    x_to_plot = np.arange(x_plus_min, x_minus_max, dx)

    def rho_sol_combined(x):
        if x_plus_1 <= x and x <= x_minus_1:
            return sol_rho_1(x)
    
        elif x_plus_2 <= x - (x_minus_1 - x_plus_2) and x - (x_minus_1 - x_plus_2) <= x_minus_2:
            return sol_rho_2(x-(x_minus_1 - x_plus_2))

    def u_sol_combined(x):
        if x_plus_1 <= x and x <= x_minus_1:
            return sol_u_1(x)
    
        elif x_plus_2 <= x - (x_minus_1 - x_plus_2) and x - (x_minus_1 - x_plus_2) <= x_minus_2:
            return sol_u_2(x-(x_minus_1 - x_plus_2))

    rho_sol_combined = np.vectorize(rho_sol_combined)
    u_sol_combined = np.vectorize(u_sol_combined)

    def rho_per(x):
        interval = x_minus_2+(x_minus_1 - x_plus_2) - x_plus_1
        x_per = (x - x_plus_1) % interval + x_plus_1
        return rho_sol_combined(x_per)

    def u_per(x):
        interval = x_minus_2+(x_minus_1 - x_plus_2) - x_plus_1
        x_per = (x - x_plus_1) % interval + x_plus_1
        return u_sol_combined(x_per)

    rho_0 = rho_per(x_to_plot)
    u_0 = u_per(x_to_plot)
    y_0 = rho_0 * (u_0 + h(rho_0))

    Q_0_ = np.zeros([2, len(x_to_plot)])
    Q_0_[0] = rho_0
    Q_0_[1] = y_0

    return Q_0_, x_to_plot