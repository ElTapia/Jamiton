import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Importa clase solver
from solver import *
from init_conditions import *
from functions_new import *


# Parámetros
xl = 0
xr = 3000 #200 
#dx = 5
N = 160
tau= 5

rho_0 = 0.1*rhomax
u_0 = 5
#y_0 = y_u(rho_0, u_0, h)
#Q_0 = [rho_0, y_0]


#x = np.linspace(xl, xr, int((xr-xl)//dx))

#Q_0_ = Q_0_1(x, h)
#Q_0_ = Q_0_2(x, h)
#Q_0_ = Q_0_4(x, h)

#rho_init = 0.3
#Q_0_ = Q_0_3(x, h, rho_init)
#Q_0_ = Q_0_5(x, h, rho_init)
#Q_0_ = Q_0_6(x, h, rho_init)
#rho_s = 0.55
# Q_0_, x, teo_rho, teo_u = Q_0_jam(h, N, tau, rho_s)

#rho_s_1 = 0.515
#rho_s_2 = 0.551
#Q_0_, x = Q_0_collide(h, N, tau, rho_s_1, rho_s_2)

#rho_izq = 0.3*rhomax
#u_izq = 30
#y_izq = y_u(rho_izq, u_izq, h)

#sol = ARZ_periodic(F_HLL, Q_0_, N, x, U, h, tau, teo_rho, teo_u)
#sol = ARZ_periodic(F_HLL, Q_0_, N, x, U, h, tau)
#sol = ARZ_periodic(F_teo, Q_0_, dx, x, U, h, tau, teo_rho, teo_u, 1)

def comparative(rho_s, error=False):
    Q_0_, x, teo_rho, teo_u = Q_0_jam(h, N, tau, rho_s)
    sol = ARZ_periodic(F_HLL, Q_0_, N, x, U, h, tau, teo_rho, teo_u, error=error)
    plt.show()

def collide(rho_s_1, rho_s_2, N, x_init=None):
    Q_0_, x = Q_0_collide(h, N, tau, rho_s_1, rho_s_2, x_init)
    sol = ARZ_periodic(F_HLL, Q_0_, N, x, U, h, tau)
    plt.show()

rho_s_1 = 0.43333743795471785
rho_s_2 = 0.305549650098795
#comparative(0.55, True)
collide(rho_s_1, rho_s_2, N)
# 26.60196099973708
