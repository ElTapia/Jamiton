import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Importa clase solver
from solver import *
from init_conditions import *


# Parámetros
xl = -2000
xr = -xl
dx = 20
tau= 5

sol = Periodic_ARZ(Q_0_3, dx, xl, xr, U, tau)
plt.show()