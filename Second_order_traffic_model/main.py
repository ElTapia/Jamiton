import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Importa clase solver
from solver import *
from init_conditions import *
from functions_new import *


# Parámetros
xl = 0
xr = 200 #5_000
dx = 3
tau= 5

rho_0 = 0.1*rhomax
u_0 = 5
y_0 = y_u(rho_0, u_0, h)
Q_0 = [rho_0, y_0]

x = np.linspace(xl, xr, (xr-xl)//dx)


#Q_0_ = Q_0_1(x, h)
#Q_0_ = Q_0_2(x, h)
#Q_0_ = Q_0_4(x, h)

rho_init = 0.3
#Q_0_ = Q_0_3(x, h, rho_init)
Q_0_ = Q_0_5(x, h, rho_init)
#Q_0_[0] = Q_0_[0]

#rho_izq = 0.4
#u_izq = 8
#y_izq = y_u(rho_izq, u_izq, U)

sol = ARZ_periodic(Q_0_, dx, xl, xr, U, h, tau)  #ARZ_infinite(Q_0, dx, xl, xr, U, tau, [rho_izq, y_izq])
plt.show()