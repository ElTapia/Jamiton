import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import root
from scipy.interpolate import interp1d

# Parámetros
gamm_1 = 1/5
gamma_2 = 1/10
beta = 12 #8
umax = 20 #19.2 #20
rhomax = 1/7.5
c = 0.078 * umax * rhomax
b = 1/3
l = 1/10

# Define u en función de rho e y
def u(rho, y, h):
    output = y/rho - h(rho)
    return output


# Define y en función de rho y u
def y_u(rho, u, h):
    output = rho*(u + h(rho))
    return output


# Flujo del modelo
# TODO: Verificar que funciona
def flux(Q, h):
    
    # Rescata variables
    rho, y = Q
    
    # Obtiene u en funcion de rho e y
    #u_ = u(rho, y, h)

    return np.array([y-rho*h(rho), (y**2)/rho - h(rho)*y]) #np.array([rho * u_, y * u_])


# Flujo de godunov teórico
def F_teo(Q, N, U, h):

    # Guarda flujo en un arreglo
    F_ = np.zeros(Q.shape)

    for i in range(1, N-1):

        # Rescata actual y vecinos
        Q_left = Q[:, i-1]
        Q_i = Q[:, i]
        Q_right = Q[:, i+1]

        # Problema de Riemann en cada vecino
        w_left = w(Q_left, Q_i, U, h)
        w_right = w(Q_i, Q_right, U, h)

        # Evalúa en el flujo del modelo
        F_[:, i] = flux(w_right, h) - flux(w_left, h)

    # Asume condiciones de borde periódicas
    Q_0 = Q[:, 0]
    Q_1 = Q[:, 1]
    Q_ult = Q[:, -1]
    Q_pen = Q[:, -2]

    w_right_0 = w(Q_0, Q_1, U, h)
    w_left_pen = w(Q_pen, Q_ult, U, h)

    F_[:, 0] = flux(w_right_0, h) - flux(w_left_pen, h)

    return F_

# Flujo con HLL
def F_HLL(Q, N, U, h):

    # Guarda flujo en un arreglo
    F_ = np.zeros(Q.shape)

    for i in range(1, N-1):

        # Rescata actual y vecinos
        Q_left = Q[:, i-1]
        Q_i = Q[:, i]
        Q_right = Q[:, i+1]

        # Evalúa en el flujo HLL
        F_[:, i] = flux_HLL(Q_i, Q_right, h) - flux_HLL(Q_left, Q_i, h)

    # Asume condiciones de borde periódicas
    Q_0 = Q[:, 0]
    Q_1 = Q[:, 1]
    Q_ult = Q[:, -1]
    Q_pen = Q[:, -2]

    F_[:, 0] = flux_HLL(Q_0, Q_1, h) - flux_HLL(Q_pen, Q_ult, h)

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
# Función de duda
def h(rho, rho_max=rhomax, gamma_1=gamm_1, gamma_2=gamma_2, u_max=umax):

    if type(rho) is float or type(rho) is np.float64:
        rho = np.array(rho)

    def h_aux(rho):
        rho_bar = rho/rho_max
        #output = beta * ((rho_bar**gamma_1)/((((1 - rho_bar)**2 + 1e-10**2))**(gamma_2/2)))
        output = beta * ((rho_bar**gamma_1)/((1 - rho_bar)**gamma_2))
        return output


    # Alcanza rho_max
    #if rho_max in rho:

        # Arreglo auxiliar
    #   output = np.empty_like(rho, dtype=float)

        # Busca valores donde se alcanza rhomax
    #    mask = np.isclose(rho, rho_max)

        # rho es rhomax
    #    output[mask] = u_max - U(rho[mask]) 

        #rho es distinto a rhomax
    #    output[~mask] = h_aux(rho[~mask])
        
    #    return output

    return 8*(rho/(rho_max-rho))**(1/2) #h_aux(rho) #8*(rho/((((rho_max-rho)**2 + 1e-2**2)))**(1/2))**(1/2) #h_aux(rho)


def h_prime(rho, rho_max=rhomax):
    gamma = 1/2
    beta = 8
    output =  beta * gamma * (rho /(rho_max - rho))**(gamma-1) * rho_max / ((rho_max - rho)**2)
    return output


# Función g
def g(y, b=b, l=l):
    output = np.sqrt(1 + ((y-b)/l)**2)
    return output


# Diagrama fundamental
def Q_e(rho, rho_max=rhomax, c=c):
    output = c * (g(0) + (((g(1) - g(0)) * rho/rho_max)) - g(rho/rho_max))
    return output


# Derivada de g
def g_prime(y, b=b, l=l):
    output = (1/l**2) * ((y-b)/ np.sqrt(1 + ((y-b)/l)**2)) 
    return output


# Derivada de Q
def Q_prime(rho, rho_max=rhomax, c=c):
    output = c * (((g(1) - g(0))/rho_max) - (g_prime(rho/rho_max)/rho_max))
    return output


# Velocidad de equilibrio
def U(rho, u_max=umax):
    output = Q_e(rho)/(np.sqrt((rho**2 + 1e-5**2)))
    return output


# Derivada de la velocidad de equilibrio
def U_prime(rho):
    output = (Q_prime(rho) - U(rho))/np.sqrt((rho**2 + 1e-5**2))
    return output


# Inversa de U
def U_inv_points(z, rho_max=rhomax):
    z = float(z)
    U_to_inv = lambda x: U(x)-z

    rho = np.real(root(U_to_inv, 0.5*rho_max).x[0])
    return rho

zs_U = np.linspace(umax, 0, 50)
U_inv_to_poly = [U_inv_points(z) for z in zs_U]

U_inv = interp1d(zs_U, U_inv_to_poly)


# Inversa de la derivada de Q
def Q_p_inv_points(z, rho_max=rhomax):
    z = float(z)
    Q_to_inv = lambda x: Q_prime(x)-z
    
    rho = np.real(root(Q_to_inv, 0.3*rho_max).x[0])
    return rho

zs_Q = np.linspace(-2, umax, 50)
Q_p_inv_to_poly = [Q_p_inv_points(z) for z in zs_Q]

Q_p_inv = interp1d(zs_Q, Q_p_inv_to_poly)


# Solución problema de Riemann
# TODO: Revisar
def w(Q_l, Q_r, U, h, u_max=umax, rho_max=rhomax):
    
    # Rescata variables a la izquierda
    rho_l, y_l = Q_l
    
    # Rescata variables a la derecha
    rho_r, y_r = Q_r

    # Obtiene velocidades
    u_l = u(rho_l, y_l, h)
    u_r = u(rho_r, y_r, h)  

    if u_r - u_l + U(rho_l) > u_max:
        
        l_1_l = u_l + rho_l * U_prime(rho_l)
        
        if l_1_l <= 0:
            rho_w = Q_p_inv(-u_l + U(rho_l))
            u_w = U(rho_w) + u_l - U(rho_l)

        else:
            rho_w = rho_l
            u_w = u_l
    
    elif 0 <= u_r - u_l + U(rho_l) and u_r - u_l + U(rho_l) <= u_max:
        u_0 = u_r
        rho_0 = U_inv(u_r - u_l + U(rho_l))

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

            elif l_1 < 0  and 0 <= l_1_0:
                rho_w = Q_p_inv(-u_l + U(rho_l))
                u_w = U(rho_w) + u_l - U(rho_l)

            # Podría dar problemas
            elif l_1_0 < 0:
                rho_w = rho_0
                u_w = u_0

    else:

        if u_r >= (rho_l * u_l)/rho_max:
            u_w = u_l
            rho_w = rho_l
        
        else:
            u_w = u_r
            rho_w = rho_max # podria dar problema por la elección de Q_e

    y_w = y_u(rho_w, u_w, h)

    return np.array([rho_w, y_w])


# Solver HLL para riemann
def flux_HLL(Q_l, Q_r, h):
    # Rescata variables a la izquierda
    rho_l, y_l = Q_l

    # Rescata variables a la derecha
    rho_r, y_r = Q_r

    # Obtiene velocidades
    u_l = u(rho_l, y_l, h)
    u_r = u(rho_r, y_r, h)

    l_1_l = u_r - rho_r * h_prime(rho_r)
    l_1_r = u_l - rho_l * h_prime(rho_l)
    l_2_l = u_l
    l_2_r = u_r

    s_l = np.min([l_1_l, l_1_r])
    s_r = np.max([l_2_l, l_2_r])

    s_R_plus = np.max([s_r, 0])
    s_l_minus = np.min([s_l, 0])

    F_l = flux(Q_l, h)
    F_r = flux(Q_r, h)

    F_hat = (s_R_plus * F_l - s_l_minus * F_r + s_l_minus * s_R_plus * (Q_r - Q_l))/(s_R_plus - s_l_minus)
    return F_hat


# Obtiene velocidad de onda máxima
def get_s_max():
    rho_s_zero = root(lambda rho: U_prime(rho) + h_prime(rho), 0.2*rhomax).x[0]
    v_s_zero = 1/rho_s_zero
    m_max = -h_bar_prime(v_s_zero)
    s_max = U_bar(v_s_zero) - m_max * v_s_zero
    return s_max


# Condición CFL
def cfl(dt, dx, Q, I_plus, eps=1e-2, u_max=umax, rho_max=rhomax):
    rho, y = Q
    zero_rho = rho[np.isclose(rho, 0)]
    
    u_ = u(rho, y, h)

        #l_max = np.max([np.max(u_), np.max(u_-2*rho**2/rho_max)])
    s_max = get_s_max()

        # Podría dar problemas
    new_dt = dx/(2*s_max) #dx/(2*u_max) #dx/(2*(u_max + np.max([-Q_prime(rho_max), I_plus]))) #
    
        # Condición si nuevo dt es mayor
        #if new_dt > dt and dt != 0:
        #    return dt
    
    #else:
    #    new_dt = dt

    return new_dt


############################### Variables lagrangeanas #########################################

# Funciones del modelo
def h_bar(v):
    return h(1/v)

def h_bar_prime(v):
    output = -h_prime(1/v)/v**2 
    return output

def U_bar(v):
    return U(1/v)

def w_v(v, m, s):
    output = U_bar(v) - (m * v + s)
    return output

def r(v, m):
    output = m * h_bar(v) + m**2 * v
    return output

def r_prime(v, m):
    output = m * h_bar_prime(v) + m**2
    return output

def ode_jam_v(x, v, tau, m, s):
    output = w_v(v, m, s)/(r_prime(v, m) * v * tau)
    return output

def ode_jam_v_eta(eta, v, m, s):
    output = w_v(v, m, s)/(r_prime(v, m) * v)
    return output