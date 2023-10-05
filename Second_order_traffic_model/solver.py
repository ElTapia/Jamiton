# Solver general de godunov
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import TextBox
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
        # Asigna mismos valores en toda la grilla
        self.Q = np.zeros([2, len(self.x)])
        self.Q[0] = self.Q[0] + Q_0[0] #Q_0(self.x, self.U)
        self.Q[1] = self.Q[1] + Q_0[1]
        self.dt = 0
        self.t = 0

        # Gráfico animado
        self.fig, self.axs = plt.subplots(1, 2)
        
        # Gráfico densidad
        self.axs[0].set_title('Densidad')
        self.axs[0].set_ylabel(r"$\rho$")
        self.axs[0].set_xlabel("x")
        self.axs[0].set_ylim(-0.1, 1.0)
        
        # Gráfico velocidad
        self.axs[1].set_title('Velocidad')
        self.axs[1].set_ylabel(r"$u$")
        self.axs[1].set_xlabel("x")
        self.axs[1].set_ylim(-10, 40.0)

        # Plotea lineas
        self.p_1, = self.axs[0].plot(self.x, self.Q[0], color="r")
        self.p_2, = self.axs[1].plot(self.x, self.Q[1], color="b")

        self.animation = animation.FuncAnimation(
            self.fig, self.update, frames=200, interval=1)#, blit=True)
        self.paused = False
        self.started = False

        self.fig.canvas.mpl_connect('button_press_event', self.toggle_pause)
        self.fig.canvas.mpl_connect('key_press_event', self.press_event)
    
    # Función para poner pausa
    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused
        
    
    # Detecta botones presionados
    def press_event(self, event):
        print("presionaste: ", event.key)
        if event.key == "enter":
            self.toggle_start()

        if event.key == "up":
            self.toggle_rho_up()

        if event.key == "down":
            self.toggle_rho_down()
            
        if event.key == "w":
            self.toggle_u_up()

        if event.key == "s":
            self.toggle_u_down()


    # Aumenta densidad
    def toggle_rho_up(self, *args, **kwargs):
        self.Q[0] += 0.1

    # Disminuye densidad
    def toggle_rho_down(self, *args, **kwargs):
        self.Q[0] -= 0.1
    
    # Aumenta velocidad
    def toggle_u_up(self, *args, **kwargs):
        self.Q[1] += 5

    # Disminuye velocidad
    def toggle_u_down(self, *args, **kwargs):
        self.Q[1] -= 5


    
    # Función para variar densidad y velocidad en un punto
    def toggle_start(self, *args, **kwargs):
        if not self.started:
            self.Q[0][self.N//2] += 0.4
            self.Q[1][self.N//2] -= 5     
            
        self.started = not self.started  
    
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
        self.p_1.set_ydata(self.Q[0])
        self.p_2.set_ydata(self.Q[1])
        self.axs[0].set_title('Densidad t=' + str("%.2f" % self.t))
        self.axs[1].set_title('Velocidad t=' + str("%.2f" % self.t))
        return [self.p_1, self.p_2,]

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
        