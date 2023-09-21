# Solver general de godunov
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from abc import ABC, abstractmethod

# Funciones de simulación
from functions import *

# Implementa Godunov
# Condiciones de borde se especializan en clases hijas
class ARZ(ABC):
    
    def __init__(self, Q_0, dx, xl, xr, U, tau):
        # Guarda variables
        self.dx = dx
        self.U = U
        self.tau = tau
        
        # Largo de la grilla
        self.L = xr - xl
        
        # Numero de puntos
        self.N =  int(self.L//self.dx)
    
        # Grilla
        self.x = np.linspace(xl, xr, self.N)
    
        # Condición inicial
        self.Q = Q_0(self.x, self.U)
        self.dt = 0
        self.t = 0
        
        # Gráfico animado
        fig, ax = plt.subplots()
        ax.set_title('Godunov periódico')
        ax.set_ylabel(r"$\rho$")
        ax.set_xlabel("x")
        ax.set_ylim(-0.1, 1.0)

        # Start with a normal distribution
        self.p, = ax.plot(self.x, self.Q[0], color="r")

        self.animation = animation.FuncAnimation(
            fig, self.update, frames=200, interval=50)#, blit=True)
        self.paused = False

        fig.canvas.mpl_connect('button_press_event', self.toggle_pause)
    
    # Función para poner pausa
    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused
        
    
    def update(self, i):
        
        # Actualiza segun condicion CFL
        self.dt = cfl(self.dt, self.dx, self.Q)
        self.t += self.dt
        # Lambda
        l = self.dt/self.dx
        
        # Paso de Godunov
        self.Q = self.Q - l * F(self.Q, self.N, self.U, l)
        
        # Agrega no homogeneidad
        rho_sig, y_sig = self.Q
        
        # Resuelve termino de relajación
        y_sig__ = y_sig * (1 - self.dt/(2 *self.tau * rho_sig))
        y_sig_ = y_sig - (self.dt * y_sig__)/(self.tau * rho_sig)
        self.Q[1] = y_sig_
        
        # Agrega condiciones de borde
        self.border_conditions()
        
        # Actualiza gráfico
        self.p.set_ydata(self.Q[0])
        return [self.p,]

    # Condiciones de borde
    @abstractmethod
    def border_conditions(self):
        pass


# ARZ con condiciones de borde periódicas
class ARZ_periodic(ARZ):
    
    def __init__(self, Q_0, dx, xl, xr, U, tau):
        
        # Init clase padre
        super().__init__(Q_0, dx, xl, xr, U, tau)
    
    # Especializa condiciones de borde
    def border_conditions(self):
        self.Q[:, 0] = self.Q[:, -2]
        self.Q[:, -1] = self.Q[:, 1]


# ARZ con borde Dirichlet y Neumann
class ARZ_infinite(ARZ):
    
    def __init__(self, Q_0, dx, xl, xr, U, tau, Q_izq):
        
        # Init clase padre
        super().__init__(Q_0, dx, xl, xr, U, tau)
        self.Q_izq = Q_izq
    
    # Especializa condiciones de borde
    def border_conditions(self):
        
        # Condición Dirichlet a la izquierda
        # Entran autos por un tiempo
        if 0<self.t and self.t<50:
            # Densidad
            self.Q[0][0] = self.Q_izq[0]
        
            # Velocidad
            self.Q[1][0] = self.Q_izq[1]
        
        # Entran menos autos
        else:
            # Densidad
            self.Q[0][0] = 0.1
        
            # Velocidad
            self.Q[1][0] = 10
        
        # Neumann lado derecho
        self.Q[:, -1] = self.Q[:, -2]
        