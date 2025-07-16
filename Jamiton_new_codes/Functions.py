from Config import *

# Base parameters
gamma = 1/2
beta = 8
u_max = 20
rho_max = 1/7.5

c = 0.078 * u_max * rho_max
b = 1/3
l = 1/10


def h(rho):
    output = beta * (rho /(rho_max - rho))**gamma
    return output

def h_prime(rho):
    output =  beta * gamma * (rho /(rho_max - rho))**(gamma-1) * rho_max / ((rho_max - rho)**2)
    return output

def g(y):
    output = np.sqrt(1 + ((y-b)/l)**2 )
    return output

def g_prime(y):
    output = (1/l**2) * ((y-b)/ np.sqrt(1 + ((y-b)/l)**2)) 
    return output

def Q_e(rho):
    output = c * (g(0) + (((g(1) - g(0)) * rho/rho_max)) - g(rho/rho_max))
    return output

def Q_prime(rho):
    output = c * ( ((g(1) - g(0))/rho_max) - (g_prime(rho/rho_max)/rho_max))
    return output

# Velocidad de equilibrio
def U(rho):
    output = Q_e(rho)/rho
    return output

# Derivada de la velocidad de equilibrio
def U_prime(rho):
    output = (Q_prime(rho) - U(rho))/np.sqrt((rho**2 + 1e-5**2))
    return output

# Funciones del modelo
def h_bar(v):
    return h(1/v)

def h_bar_prime(v):
    output = -h_prime(1/v)/(v**2) 
    return output

def U_bar(v):
    return U(1/v)

def U_bar_prime(v):
    return -U_prime(1/v)/(v**2)

def w_v(v, m, s):
    output = U_bar(v) - (m * v + s)
    return output

def w_v_prime(v, m, s):
    output = U_bar_prime(v) - m
    return output

def r(v, m):
    output = m * h_bar(v) + m**2 * v
    return output

def r_prime(v, m):
    output = m * h_bar_prime(v) + m**2
    return output

# Transforma rho a u
def rho_to_u(rho, m, s):
    u = (m/rho) + s
    return u

# Transforma v a rho
def v_to_rho(v):
    rho = 1/v
    return rho

def ode_jam_v(x, v, tau, m, s):
    output = w_v(v, m, s)/(r_prime(v, m) * v * tau)
    return output

def ode_jam_v_eta(eta, v, m, s):
    output = w_v(v, m, s)/(r_prime(v, m) * v)
    return output
