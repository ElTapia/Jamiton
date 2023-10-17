import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import newton
from scipy.interpolate import interp1d

# Parámetros
umax = 20
gamm = 0.5 
b = 8
rhomax = 1/7.5

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


# Funciones del modelo
# Funcion de duda
def h(rho, beta=b, rho_max=rhomax, gamma=gamm):
    output = beta * (rho /(rho_max - rho))**gamma
    return output


# Velocidad de equilibrio
def U(rho, u_max=umax):
    return u_max - h(rho)


# Derivada de U
def U_prime(rho, rho_max=rhomax, gamma=gamm):
    output = gamma * rho_max * h(rho) * (rho/(rho_max - rho))
    return output


# Diagrama fundamental
def Q_e(rho, rho_max=rhomax):
    if np.isclose(rho, rho_max):
        return 0
    
    output = rho * U(rho)
    return output


# Funcion inversa de U
def U_inv(z, u_max=umax, beta=b, rho_max=rhomax, gamma=gamm):
    output = rho_max * (((u_max-z)/beta)**(1/gamma)/(1 + ((u_max-z)/beta)**(1/gamma)))
    return output


# Derivada de Q
def Q_prime(rho, rho_max=rhomax, gamma=gamm):
    output = U(rho) + gamma * rho_max * h(rho) * (rho**2)/(rho_max - rho)
    return output


# Calcula inversa de la derivada de Q
# Función para pocos puntos
def Q_p_inv_points(z, u_max=umax, rho_max=rhomax):
    z = float(z)
    Q_to_root = lambda x: Q_prime(x)-z
    
    if 19.7 < z <= u_max:
        rho = np.real(newton(Q_to_root, 0.0))
        return rho
    
    rho = np.real(newton(Q_to_root, 0.8*rho_max))
    return rho


# Evalua en algunos puntos
zs = np.linspace(0.96, umax, 50)
Q_p_inv_to_poly = [Q_p_inv_points(z) for z in zs]

# Interpola para obtener función inversa
Q_p_inv = interp1d(zs, Q_p_inv_to_poly)

# Funciona como el rho_max
root_U = np.real(newton(U, rhomax-1e-4)) 


# Solución problema de Riemann
def w(Q_l, Q_r, U, u_max=umax, gamma=gamm, rho_max=rhomax):
    
    # Rescata variables a la izquierda
    rho_l, y_l = Q_l
    
    # Rescata variables a la derecha
    rho_r, y_r = Q_r
      
    # Obtiene velocidades
    u_l = u(rho_l, y_l, U)
    u_r = u(rho_r, y_r, U)  

    if u_r - u_l + U(rho_l) > u_max:
        
        l_1_l = u_l + rho_l * U_prime(rho_l)
        
        if l_1_l <= 0:
            rho_w = Q_p_inv(-u_l + U(rho_l))
            u_w = U(rho_w) + u_l - U(rho_l)
        
        else:
            rho_w = rho_l
            u_w = u_l
    
    elif 0<= u_r - u_l + U(rho_l) and u_r - u_l + U(rho_l) < -u_max:
        u_0 = u_r
        rho_0 = U_inv(u_r-u_l+U(rho_l))

        if u_r <= u_l:
            q_0 = rho_0 * u_r
            q_l = rho_l * u_l
            
            if q_0 <= q_l:
                rho_w = rho_0
                u_w = u_0
            
            else:
                rho_w = rho_l
                u_w = u_l
        else:
            l_1   = u_l + rho_l * U_prime(rho_l)
            l_1_0 = u_0 + rho_0 * U_prime(rho_0)

            if l_1 >= 0:
                rho_w = rho_l
                u_w = u_l

            elif l_1 < 0  and 0<= l_1_0:
                rho_w = Q_p_inv(-u_l + U(rho_l))
                u_w = U(rho_w) + u_l - U(rho_l)

            # Podría dar problemas
            elif l_1_0 < 0:
                rho_w = rho_0
                u_w = u_0

    else:
        if u_r >= (rho_l * u_l)/root_U: #rho_max
            u_w = u_l
            rho_w = rho_l
        
        else:
            u_w = u_r
            rho_w = root_U #rho_max # podria dar problema por la elección de Q

    y_w = y_u(rho_w, u_w, U)

    return np.array([rho_w, y_w])



# Condición CFL
def cfl(dt, dx, Q, eps=1e-2, u_max=umax, rho_max=rhomax):
    rho, y = Q
    zero_rho = rho[np.isclose(rho, 0)]

    # No hay densidad 0
    if len(zero_rho) == 0:
        u_ = u(rho, y, U)
    
        #l_max = np.max(u_)
        Q_e_list = np.array([Q_e(rh) for rh in rho])
        I_max = np.max(np.fabs(rho * u_ - Q_e_list))

        # Podría dar problemas
        new_dt = dx/(2* (u_max + np.max([Q_prime(root_U), I_max]))) #dx/(2*l_max)
    
        # Condición si nuevo dt es mayor
        if new_dt < dt and dt != 0:
            return dt
    
    else:
        new_dt = dt

    print(new_dt)
    return new_dt


