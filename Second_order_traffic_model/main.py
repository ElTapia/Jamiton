import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Importa clase solver
from solver import *
from init_conditions import *
from functions import *


# Parámetros
xl = -2000
xr = -xl
dx = 20
tau= 5

rho_0 = 0.1
u_0 = 10
y_0 = y_u(rho_0, u_0, U)
Q_0 = [rho_0, y_0]

rho_izq = 0.4
u_izq = 8
y_izq = y_u(rho_izq, u_izq, U)

sol = ARZ_periodic(Q_0, dx, xl, xr, U, tau)  #ARZ_infinite(Q_0, dx, xl, xr, U, tau, [rho_izq, y_izq])
plt.show()