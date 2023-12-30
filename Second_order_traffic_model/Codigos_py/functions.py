import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Parámetros
u_max = 102#20.8 #20.8 #2 92
gamm = 0.027#0.03 #0.03

# Define u en función de rho e y
def u(rho, y, U):
    output = y/rho + U(rho)
    return output


# Define y en función de rho y u
def y_u(rho, u, U):
    output = rho*(u - U(rho))
    return output


# Flujo del modelo
def flux(Q, U):
    
    # Rescata variables
    rho, y = Q
    
    # Obtiene u en funcion de rho e y
    u_ = u(rho, y, U)
    return np.array([rho * u_, y * u_])


# Flujo de Godunov de primer orden
def F(Q, N, U, l):
    
    # Guarda flujo en un arreglo
    F_ = np.zeros(Q.shape)
    
    for i in range(1, N-1):
        
        # Rescata actual y vecinos
        Q_left = Q[:, i-1]
        Q_i = Q[:, i]
        Q_right = Q[:, i+1]
        
        # Problema de Riemann en cada vecino
        w_left = w(Q_left, Q_i, U)
        w_right = w(Q_i, Q_right, U)
        
        # Evalúa en el flujo del modelo
        F_[:, i] = flux(w_right, U) - flux(w_left, U)
    
    return F_


# Integral de densidad
def density_integral(x, dx, N_t, Q):
    integral_graph = []
    for n in range(N_t):
    
        density = Q[n, 0, :]
        integral = integrate.simpson(density, x, dx)
        integral_graph += [integral]
    
    plt.plot(integral_graph)
    plt.show()


# Velocidad de equilibrio
def U(rho, umax=u_max, gamma = gamm):
    return umax*(1-rho**gamma)


# Solución problema de Riemann homogeneo
# TODO: Quitar casos redundantes
# TODO: Cambiar si se quiere cambiar el h(rho) o U(rho)
# TODO: Agregar Riemann no homogeneo
# Solución problema de Riemann homogeneo

# TODO: Quitar casos redundantes
# TODO: Cambiar si se quiere cambiar el h(rho) o U(rho)
# TODO: Agregar Riemann no homogeneo
def w(Q_l, Q_r, U, umax=u_max, gamma=gamm):
    
    # Rescata variables a la izquierda
    rho_l, y_l = Q_l
    
    # Rescata variables a la derecha
    rho_r, y_r = Q_r
      
    # Obtiene velocidades
    u_l = u(rho_l, y_l, U)
    u_r = u(rho_r, y_r, U)
    u_m = u_r

    # Define valor medio de rho y u
    rho_m = (rho_l**gamma + (u_l - u_r)/umax)**(1/gamma)
    
    if umax*rho_l**gamma + u_l <= u_r:
        print("No hay solución")
        print("u_r = ", u_r, "demás = ", umax*rho_l**gamma + u_l)

    # u_r y u_l cercanos
    if np.isclose(u_r, u_l):

        # Solucion en 0
        rho_0 = rho_l
        u_0 = u_l

    # u_r menor a u_l
    elif u_r < u_l:

        # Define velocidad del shock
        l_s = (rho_m * u_m - rho_l * u_l)/(rho_m - rho_l)

        # Si es positivo
        if l_s >= 0:

            # Solucion en 0
            rho_0 = rho_l
            u_0 = u_l

        # Si es negativo
        else:

            # Solucion en 0
            rho_0 = rho_m
            u_0 = u_m

    # Condición u_l en un intervalo en función de u_r
    elif u_r - umax*rho_l**gamma < u_l < u_r:

        # Lambdas a la izquierda y centrado con respecto al 0
        l_0_l = u_l - umax * gamma * rho_l**gamma
        l_0_m = u_r - umax*gamma*rho_l**gamma + gamma*(u_r - u_l)

        # Lambda izquierda es positivo
        if l_0_l >= 0:

            # Solucion en 0
            rho_0 = rho_l
            u_0 = u_l

        # Lambda derecha es negativo
        elif l_0_m <= 0:

            # Solucion en 0
            rho_0 = rho_m
            u_0 = u_m

        # A la izquierda es negativo y centrado positivo
        elif l_0_l < 0 < l_0_m:
            rho_bar = ((u_l + umax*rho_l**gamma)/((gamma + 1)*umax))**(1/gamma)
            u_bar = (gamma/(gamma+1)) * (u_l + umax*rho_l**gamma)

            # Solucion en 0
            rho_0 = rho_bar
            u_0 = u_bar

    # u_l menor a u_r menos algo
    elif u_l <= u_r - umax*rho_l**gamma:

        # Lambda a la izquierda del 0
        l_0_l = u_l - umax * gamma * rho_l**gamma

        # Si es positivo
        if l_0_l >= 0:

            # Solucion en 0
            rho_0 = rho_l
            u_0 = u_l

        # Si es negativo
        else:
            rho_bar = ((u_l + umax*rho_l**gamma)/((gamma + 1)*umax))**(1/gamma)
            u_bar = (gamma/(gamma+1)) * (u_l + umax*rho_l**gamma)

            # Solucion en 0
            rho_0 = rho_bar
            u_0 = u_bar

    # rho_l cero y rho_r positivo
    # TODO: Ver equivalencia a algun caso anterior
    elif np.isclose(rho_l, 0) and rho_r >0:
        rho_0 = 0
        u_0 = 0
    
    # Opuesto al anterior
    # TODO: Ver equivalencia a algun caso anterior
    elif rho_l > 0 and np.isclose(rho_r, 0):
        
        # Velocidad
        u_l = u(rho_l, y_l, U)
        
        # Obtiene lambda a la izquierda del 0
        l_0_l = u_l - umax * gamma * rho_l**gamma
        
        # Si es positivo
        if l_0_l >= 0:
            
            # Solucion en 0
            rho_0 = rho_l
            u_0 = u_l
        
        # Si es negativo
        else:
            rho_bar = ((u_l + umax*rho_l**gamma)/((gamma + 1)*umax))**(1/gamma)
            u_bar = (gamma/(gamma+1)) * (u_l + umax*rho_l**gamma)
            
            # Solucion en 0
            rho_0 = rho_bar
            u_0 = u_bar
    
    # rho_l  es cero
    # TODO: Escribir casos densidad = 0
    else: #np.isclose(rho_l, 0):
        rho_0 = 0
        u_0 = 0
        
    
    # Obtiene y_0 segun el u_0 obtenido
    y_0 = y_u(rho_0, u_0, U)
    
    return np.array([rho_0, y_0])


#def w_(Q_l, Q_r, U, Vmax=V_max, rhocr=rho_cr, rhomax=rho_max, Wmax=W_max, Qmax=Q_max):
#    pass


# Condición CFL
def cfl(dt, dx, Q, eps=1e-2):
    rho, y = Q
    zero_rho = rho[np.isclose(rho, 0)]
    
    # Hay densidad 0
    if len(zero_rho) == 0:
        u_ = u(rho, y, U)
    
        l_max = np.max(u_)
    
        new_dt = dx/(2*l_max)
    
        # Condición si nuevo dt es mayor
        if new_dt > dt and dt != 0:
            return dt
    
    else:
        new_dt = dt
    
    return new_dt


