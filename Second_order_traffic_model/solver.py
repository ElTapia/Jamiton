# Solver general de godunov
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import TextBox
from abc import ABC, abstractmethod

# Funciones de simulación
from functions_new import *

# Implementa Godunov
# Condiciones de borde se implementan en clases hijas
class ARZ(ABC):
    
    def __init__(self, Q_0, dx, xl, xr, U, h, tau):
        # Guarda variables
        self.dx = dx
        self.U = U
        self.h = h
        self.tau = tau
        
        # Largo de la grilla
        self.L = xr - xl
        
        # Numero de puntos
        self.N =  int(self.L//self.dx)
    
        # Grilla
        self.x = np.linspace(xl, xr, self.N)
    
        # Condición inicial
        # Asigna mismos valores en toda la grilla
        self.Q_0 = Q_0
        self.Q = np.zeros([2, self.N])#len(self.x)])
        
        # Asigna valor inicial homogéneo
        self.Q[0] = self.Q[0] + self.Q_0[0] #Q_0(self.x, self.U)
        self.Q[1] = self.Q[1] + self.Q_0[1]
        
        # Tiempo inicial 
        self.dt = 0 # Después se actualiza al valor que corresponde
        self.t = 0

        # Gráfico animado
        self.fig = plt.figure(figsize=(12, 8))
        self.gs = self.fig.add_gridspec(nrows=13, ncols=13)
        ax1 =  self.fig.add_subplot(self.gs[0:11, 0:8])
        ax2 = self.fig.add_subplot(self.gs[0:11, 9:13])
        slider_ax = self.fig.add_subplot(self.gs[12:13, 2:11])
        
        self.axs = [ax1, ax2, slider_ax]
        
        # Gráfico densidad
        self.axs[0].set_title('Densidad')
        self.axs[0].set_ylabel(r"$\rho$")
        self.axs[0].set_xlabel("x")
        self.axs[0].set_ylim(-0.1, 1.0)
        #self.axs[0].set_xlim(0, 3_000)
        
        # Gráfico velocidad
        self.axs[1].set_title('Velocidad')
        self.axs[1].set_ylabel(r"$u$")
        self.axs[1].set_xlabel("x")
        self.axs[1].set_ylim(-10, 80.0)
        #self.axs[1].set_xlim(0, 3_000)


        # Plotea lineas
        self.p_1, = self.axs[0].plot(self.x, self.Q[0]/rhomax, color="r")
        self.p_2, = self.axs[1].plot(self.x, u(self.Q[0], self.Q[1], U), color="b")

        self.animation = animation.FuncAnimation(
            self.fig, self.update, frames=50, interval=1)#, blit=True)
        self.paused = False
        self.started = False

        self.fig.canvas.mpl_connect('key_press_event', self.press_event)

        self.rho_per, self.u_per = 0, 0
        self.text_box = TextBox(self.axs[2], 'Perturbación', initial="0, 0")
        self.text_box.on_submit(self.submit)


    # Ingresa perturbación inicial
    def submit(self, text):
        self.rho_per, self.u_per = eval(text)


    # Función para poner pausa
    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()

        else:
            self.animation.pause()
        self.paused = not self.paused
        
    
    # Detecta botones presionados
    def press_event(self, event):
        if event.key == "enter":
            self.toggle_start()

        if event.key == "up":
            self.toggle_rho_up()

        if event.key == "down":
            self.toggle_rho_down()
            
        if event.key == "e":
            self.toggle_u_up()

        if event.key == "d":
            self.toggle_u_down()
            
        if event.key == "r":
            self.toggle_reset()
        
        if event.key == "p":
            self.toggle_pause()


    # Aumenta densidad
    def toggle_rho_up(self, *args, **kwargs):
        self.Q[0] += 0.05*rhomax

    # Disminuye densidad
    def toggle_rho_down(self, *args, **kwargs):
        self.Q[0] -= 0.05*rhomax
    
    # Aumenta velocidad
    def toggle_u_up(self, *args, **kwargs):
        self.Q[1] += 0.5

    # Disminuye velocidad
    def toggle_u_down(self, *args, **kwargs):
        self.Q[1] -= 0.5

    # Reinicia simulación
    def toggle_reset(self, *args, **kwargs):
        self.Q[0] = self.Q_0[0]
        self.Q[1] = self.Q_0[1]
        self.t = 0
        self.started = not self.started
        self.dt=0


    # Función para empezar simulación
    def toggle_start(self, *args, **kwargs):
        if not self.started:
            self.Q[0][self.N//4] += self.rho_per*rhomax
            self.Q[1][self.N//4] += self.u_per 
            
        self.started = not self.started  

    def update(self, i):
        
        # No actualiza
        if not self.started:
            self.p_1.set_ydata(self.Q[0]/rhomax)
            self.p_2.set_ydata(u(self.Q[0], self.Q[1], U))
            return [self.p_1, self.p_2,]

        # Actualiza segun condicion CFL
        self.dt = cfl(self.dt, self.dx, self.Q)
        self.t += self.dt
        # Lambda
        l = self.dt/self.dx

        # Paso de Godunov
        self.Q = self.Q - l * F(self.Q, self.N, self.U, l)

        # Agrega no homogeneidad
        rho_sig, y_sig = self.Q
        alpha = self.dt/self.tau

        # Resuelve termino de relajación
        # TODO: Cambiar a modelo ARZ de Seibold
        #y_sig__ = y_sig * (1 - self.dt/(2 *self.tau * rho_sig))
        #y_sig_ = y_sig - ((self.dt * y_sig__)/(self.tau * rho_sig))
        y_sig_ = y_sig * (1 - alpha) + alpha * (rho_sig * self.U(rho_sig) + self.h(rho_sig))
        
        self.Q[1] = y_sig_

        # Agrega condiciones de borde
        self.border_conditions()

        # Actualiza gráfico
        self.p_1.set_ydata(self.Q[0]/rhomax)
        self.p_2.set_ydata(u(self.Q[0], self.Q[1], U))
        self.axs[0].set_title('Densidad t=' + str("%.2f" % self.t))
        self.axs[1].set_title('Velocidad t=' + str("%.2f" % self.t))
        return [self.p_1, self.p_2,]

    # Condiciones de borde
    @abstractmethod
    def border_conditions(self):
        pass


# ARZ con condiciones de borde periódicas
class ARZ_periodic(ARZ):
    
    def __init__(self, Q_0, dx, xl, xr, U, h, tau):
        
        # Init clase padre
        super().__init__(Q_0, dx, xl, xr, U, h, tau)
    
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
        #if 0<self.t and self.t<50:

        # Densidad
        self.Q[0][0] = self.Q_izq[0]

        # y
        self.Q[1][0] = self.Q_izq[1]

        # Entran menos autos
        #else:
        #    rho_else = 0.1
        #    u_else = 10
        #    y_else = y_u(rho_else, u_else, U)

            # Densidad
        #    self.Q[0][0] = rho_else

            # y
        #    self.Q[1][0] = y_else

        # Neumann lado derecho
        self.Q[:, -1] = self.Q[:, -2]
