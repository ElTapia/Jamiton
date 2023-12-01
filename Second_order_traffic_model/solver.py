# Solver general de godunov
import numpy as np
import scipy as sp
import scipy.sparse as spsp
import scipy.sparse.linalg as spl

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import TextBox

from abc import ABC, abstractmethod

# Funciones de simulación
from functions_new import *

# Implementa Godunov
# Condiciones de borde se implementan en clases hijas
class ARZ(ABC):

    def __init__(self, F, Q_0, N, x, U, h, tau, rho_teo=None, u_teo=None, viscosity=None, error=False):
        # Guarda variables
        #self.dx = dx
        self.U = U
        self.h = h
        self.tau = tau
        self.F = F

        # Largo de la grilla
        self.L = x[-1] - x[0] 

        # Numero de puntos
        self.N = N  # int(self.L//self.dx)
        self.dx = self.L/self.N

        # Epsilon para difusion
        self.viscosity = viscosity

        # Grillas
        self.x = x #np.linspace(xl, xr, self.N)
        #self.x_teo = np.linspace(x[0], x[-1], 1000)
    
        # Condición inicial
        self.Q_0 = Q_0
        self.Q = np.zeros([2, self.N])

        # Asigna valor inicial
        self.Q[0] += self.Q_0[0]
        self.Q[1] += self.Q_0[1]

        # Velocidad relativa inicial
        # Sirve para condición CFL
        self.I_init = u(self.Q_0[0], self.Q_0[1], self.h) - self.U(self.Q_0[0])
        self.I_plus = np.max(np.fabs(self.I_init))

        # Tiempo inicial 
        self.dt = 0 # Después se actualiza al valor que corresponde
        self.t = 0
        self.i = 0
        self.t_list = [self.t]

        # Solución analítica (si es que existe)
        self.rho_teo = rho_teo
        self.u_teo = u_teo
        self.error = error
        self.error_rho = None
        self.error_u = None

        # Gráfico animado
        self.fig = plt.figure(figsize=(12, 8))
        self.gs = self.fig.add_gridspec(nrows=13, ncols=13)
        ax1 =  self.fig.add_subplot(self.gs[0:9, 0:8])
        ax2 = self.fig.add_subplot(self.gs[0:9, 9:13])
        ax3 = self.fig.add_subplot(self.gs[10:13, 2:11])

        self.axs = [ax1, ax2, ax3]

        # Gráfico densidad
        self.axs[0].set_title('Densidad')
        self.axs[0].set_ylabel(r"$\rho/\rho_{max}$")
        self.axs[0].set_xlabel("x")
        self.axs[0].set_ylim(-0.1, 1.0)
        #self.axs[0].set_xlim(0, 3_000)

        # Gráfico velocidad
        self.axs[1].set_title('Velocidad')
        self.axs[1].set_ylabel(r"$u$")
        self.axs[1].set_xlabel("x")
        self.axs[1].set_ylim(0, 25)
        #self.axs[1].set_xlim(0, 3_000)

        # Gráfico error
        self.axs[2].set_title('Error relativo')
        self.axs[2].set_ylabel("Error")
        self.axs[2].set_xlabel("t")
        self.axs[2].set_xlim(-0.5, 60)
        self.axs[2].set_ylim(-0.05, 0.05)


        # Plotea lineas
        self.p_1, = self.axs[0].plot(self.x, (self.Q[0]/rhomax), color="r", label="Simulación")
        self.p_2, = self.axs[1].plot(self.x, u(self.Q[0], self.Q[1], self.h), color="b", label="Simulación")


        # Linea vacía
        empty_line = np.full(len(self.x), fill_value=None)
        self.p_1_teo, = self.axs[0].plot(self.x, empty_line, color="purple", ls="--", label="Teórica")
        self.p_2_teo, = self.axs[1].plot(self.x, empty_line, color="purple", ls="--", label="Teórica")
        self.axs[1].hlines(umax, self.x[0], self.x[-1], ls="--", label="u_max")


        # gráfico vacío para el error
        empty_error = np.full(len(self.t_list), fill_value=None)
        self.p_1_error, = self.axs[2].plot(self.t_list, empty_error, label="Error densidad")
        self.p_2_error, = self.axs[2].plot(self.t_list, empty_error, label="Error velocidad")

        self.axs[0].legend()
        self.axs[1].legend()
        self.axs[2].legend()

        # Plotea si hay solución analítica
        if self.rho_teo is not None and self.u_teo is not None:
            self.p_1_teo.set_ydata(self.rho_teo(self.x, 0)/rhomax)
            self.p_2_teo.set_ydata(self.u_teo(self.x, 0))

        # Plotea si hay error
        if self.error:
            self.error_rho_list = []
            self.error_u_list = []

            error_rho = np.linalg.norm(self.Q[0] - self.rho_teo(self.x, self.t), ord=1)

            u_sim = u(self.Q[0], self.Q[1], self.h)
            u_teo_error = self.u_teo(self.x, self.t)
            error_u = np.linalg.norm(u_sim - u_teo_error, ord=1)
            
            norm_teo_rho = np.linalg.norm(self.rho_teo(self.x, self.t), ord=1)
            norm_teo_u = np.linalg.norm(u_teo_error, ord=1)

            self.error_rho_list += [error_rho/norm_teo_rho]
            self.error_u_list += [error_u/norm_teo_u]

            self.p_1_error.set_data(self.t_list[0], self.error_rho_list[0])
            self.p_2_error.set_data(self.t_list[0], self.error_u_list[0])


        self.animation = animation.FuncAnimation(
            self.fig, self.update, frames=50, interval=1)#, blit=True)
        self.paused = False
        self.started = False

        self.fig.canvas.mpl_connect('key_press_event', self.press_event)

        #self.rho_per, self.u_per = 0, 0
        #self.text_box = TextBox(self.axs[2], 'Perturbación', initial="0, 0")
        #self.text_box.on_submit(self.submit)


    # Ingresa perturbación inicial
    #def submit(self, text):
    #    self.rho_per, self.u_per = eval(text)


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
        self.Q[1] += 0.2

    # Disminuye velocidad
    def toggle_u_down(self, *args, **kwargs):
        self.Q[1] -= 0.2

    # Reinicia simulación
    def toggle_reset(self, *args, **kwargs):
        self.Q[0] = self.Q_0[0]
        self.Q[1] = self.Q_0[1]
        self.t = 0
        self.dt=0
        self.started = not self.started


    # Función para empezar simulación
    def toggle_start(self, *args, **kwargs):
        #if not self.started:
        #    self.Q[0][self.N//4] += self.rho_per*rhomax
        #    self.Q[1][self.N//4] += self.u_per 

        self.started = not self.started  


    def update(self, i):

        # No actualiza
        if not self.started:
            self.p_1.set_ydata(self.Q[0]/rhomax)
            self.p_2.set_ydata(u(self.Q[0], self.Q[1], self.h))
            return [self.p_1, self.p_2,]

        # Actualiza segun condicion CFL
        self.dt = cfl(self.dt, self.dx, self.Q, self.I_plus)
        self.t += self.dt
        self.i += 1
        self.t_list += [self.t]

        # Lambda
        l = self.dt/self.dx

        # Paso de Godunov
        self.Q = self.Q - l * self.F(self.Q, self.N, self.U, self.h)


        # Resuelve termino de relajación
        if self.viscosity is not None:
            self.relaxation_term_viscous(self.viscosity)

        else:
            self.relaxation_term_inviscous()

        # Agrega condiciones de borde
        self.border_conditions()

        # Actualiza gráfico
        self.p_1.set_ydata(self.Q[0]/rhomax)
        self.p_2.set_ydata(u(self.Q[0], self.Q[1], self.h))
        self.axs[0].set_title('Densidad t=' + str("%.2f" % self.t))
        self.axs[1].set_title('Velocidad t=' + str("%.2f" % self.t))

        # Agrega solución teórica
        if self.rho_teo is not None and self.u_teo is not None:
            self.p_1_teo.set_ydata(self.rho_teo(self.x, self.t)/rhomax)
            self.p_2_teo.set_ydata(self.u_teo(self.x, self.t))

        # Agrega error
        if self.error:
            error_rho = np.linalg.norm(self.Q[0] - self.rho_teo(self.x, self.t), ord=1)

            u_sim = u(self.Q[0], self.Q[1], self.h)
            u_teo_error = self.u_teo(self.x, self.t)
            error_u = np.linalg.norm(u_sim - u_teo_error, ord=1)

            norm_teo_rho = np.linalg.norm(self.rho_teo(self.x, self.t), ord=1)
            norm_teo_u = np.linalg.norm(u_teo_error, ord=1)

            self.error_rho_list += [error_rho/norm_teo_rho]
            self.error_u_list += [error_u/norm_teo_u]

            self.p_1_error.set_data(self.t_list[:self.i], self.error_rho_list[:self.i])
            self.p_2_error.set_data(self.t_list[:self.i], self.error_u_list[:self.i])

        return [self.p_1, self.p_2, self.p_1_teo, self.p_2_teo, self.p_1_error, self.p_2_error, ]

    # Condiciones de borde
    @abstractmethod
    def border_conditions(self):
        pass

    # Término de relajación
    def relaxation_term_inviscous(self):
        # Agrega no homogeneidad
        rho_sig, y_sig = self.Q
        alpha = self.dt/self.tau
        u_sig = u(rho_sig, y_sig, self.h)

        # TODO: Agregar difusión usando método implícito
        #y_sig__ = y_sig * (1 - self.dt/(2 *self.tau * rho_sig))
        #y_sig_ = y_sig - ((self.dt * y_sig__)/(self.tau * rho_sig))
        #y_sig_ = y_sig * (1 - alpha) + alpha  * rho_sig * (self.U(rho_sig) + self.h(rho_sig)) # "jamitinos"
        #y_sig_ = alpha * (self.U(rho_sig) + self.h(rho_sig)) + y_sig * (1 - alpha/rho_sig)
        #y_sig_ = alpha * (U(rho_sig) - u_sig) + y_sig

        # Explícito
        #y_sig_ = alpha * rho_sig * (self.U(rho_sig) + self.h(rho_sig)) + (1 - alpha) * y_sig

        # Implícito
        y_sig_ = (alpha/(1+alpha)) * rho_sig * (self.U(rho_sig) + self.h(rho_sig)) + (1/(1+alpha)) * y_sig

        self.Q[1] = y_sig_


    # Agrega pequeña difusión
    def relaxation_term_viscous(self, eps):

        # Función de relajación
        def rlx_func(rho):
            return rho * (self.U(rho) + self.h(rho))

        # Parámetros
        rho, y = self.Q
        l = self.dt/(self.dx**2)    # Va cambiando con la condición CFL
        alpha = self.dt / self.tau  # Va cambiando con la condición CFL

        # Vectores de unos para diagonales
        e = np.ones(self.N) 
        f = np.ones(self.N-1)
        offset = [-1,0,1]
        I = sp.sparse.identity(self.N)

        #### Actualizacion rho ####

        # Matriz de primera actualizacion
        k_rho_1 = np.array([-f*eps*l/2, e+eps*l, -f*eps*l/2], dtype= object)
        A_rho_1 = sp.sparse.diags(k_rho_1,offset)
        A_rho_1 = A_rho_1.tolil()
        A_rho_1[0, -1] = -eps * l / 2 # Agregan periodicidad
        A_rho_1[-1, 0] = -eps * l / 2

        # Primera actualización
        B_rho_1 = 2 * I - A_rho_1
        rho_1 = sp.sparse.linalg.spsolve(A_rho_1, B_rho_1 * rho)

        # Matriz de segunda actualizacion
        k_rho_2 = np.array([-f*eps*l/3, e+2*eps*l/3, -f*eps*l/3], dtype= object) 
        A_rho_2 = sp.sparse.diags(k_rho_2, offset)
        A_rho_2 = A_rho_2.tolil()
        A_rho_2[0, -1] = -eps * l / 3 # Agregan periodicidad
        A_rho_2[-1, 0] = -eps * l / 3

        # Segunda actualización
        rho_sig = sp.sparse.linalg.spsolve(A_rho_2, (4 * rho_1 - rho)/3)

        #### Actualizacion y ####

        # Matriz de primera actualizacion
        k_y_1 = np.array([-f*eps*l/2, e+(alpha/2)+(eps*l), -f*eps*l/2], dtype= object)
        A_y_1 = sp.sparse.diags(k_y_1,offset)
        A_y_1 = A_y_1.tolil()
        A_y_1[0, -1] = -eps * l / 2 # Agregan periodicidad
        A_y_1[-1, 0] = -eps * l / 2

        # Primera actualización
        B_y_1 = 2 * I - A_y_1
        B_der_y_1 = B_y_1 * y + (alpha/2) * (rlx_func(rho) + rlx_func(rho_1))
        y_1 = sp.sparse.linalg.spsolve(A_y_1, B_der_y_1)

        # Matriz de segunda actualizacion
        k_y_2 = np.array([-f*eps*l/3, e+(2/3*eps*l)+(alpha/3), -f*eps*l/3], dtype= object) 
        A_y_2 = sp.sparse.diags(k_y_2, offset)
        A_y_2 = A_y_2.tolil()
        A_y_2[0, -1] = -eps * l / 3 # Agregan periodicidad
        A_y_2[-1, 0] = -eps * l / 3

        # Segunda actualización
        B_der_y_2 = alpha/3 * rlx_func(rho_sig) + (4 * y_1 - y)/3
        y_sig = sp.sparse.linalg.spsolve(A_y_2, B_der_y_2)

        # Asigna
        self.Q = np.array([rho_sig, y_sig])



# ARZ con condiciones de borde periódicas
class ARZ_periodic(ARZ):

    def __init__(self, F, Q_0, N, x, U, h, tau, rho_teo=None, u_teo=None, viscosity=None, error=False):

        # Init clase padre
        super().__init__(F, Q_0, N, x, U, h, tau, rho_teo, u_teo, viscosity, error)

    # Especializa condiciones de borde
    def border_conditions(self):
        self.Q[:, -1] = self.Q[:, 0]


# ARZ con borde Dirichlet y Neumann
class ARZ_infinite(ARZ):

    def __init__(self, F, Q_0, dx, x, U, h, tau, Q_izq, rho_teo=None, u_teo=None, viscosity=None, error=False):

        # Init clase padre
        super().__init__(F, Q_0, dx, x, U, h, tau, rho_teo, u_teo, viscosity, error)
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
