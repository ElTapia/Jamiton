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
dx = 5
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
Q_0_, x, dx, teo_rho, teo_u = Q_0_jam(h, tau)
#Q_0_, x, dx = Q_0_collide(h, tau)

#rho_izq = 0.3*rhomax
#u_izq = 30
#y_izq = y_u(rho_izq, u_izq, h)

sol = ARZ_periodic(F_HLL, Q_0_, dx, x, U, h, tau, teo_rho, teo_u)  # ARZ_infinite(Q_0_, dx, xl, xr, U, h, tau, [rho_izq, y_izq])
#sol = ARZ_periodic(F_teo, Q_0_, dx, x, U, h, tau, teo_rho, teo_u, 1)
plt.show()
