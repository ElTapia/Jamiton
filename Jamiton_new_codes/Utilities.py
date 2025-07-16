from Config import *
from Functions import *
import pickle
import os
import gzip

# One dimensional Newton
def newton_1D(x_0, f, f_prime, tol=1e-8):
    x_k = x_0
    error = np.add.reduce(np.abs(f(x_k)))
    while error > tol:
        f_1 = -f(x_k)
        f_2 = f_prime(x_k)
        delta_x = f_1/f_2
        x_k_1 = x_k + delta_x
        error = norm([x_k - x_k_1], 2)
        x_k = x_k_1
    if error != error:
        return 0, 1

    return x_k

# Transforma v a rho
def v_to_rho(v):
    rho = 1/v
    return rho

# Transforma rho a u
def rho_to_u(rho, m, s):
    u = (m/rho) + s
    return u


# Define u en función de rho e y
def u_y(rho, y):
    output = y/rho - h(rho)
    return output


# Define y en función de rho y u
def y_u(rho, u):
    output = rho*(u + h(rho))
    return output

# Periodic functions of rho and u
def rho_per_gen(sol_rho, x_plus, x_minus):
    def rho_per(x):
        interval = x_plus - x_minus
        return sol_rho((x - x_minus) % interval + x_minus)
    return rho_per

def u_per_gen(sol_u, x_plus, x_minus):
    def u_per(x):
        interval = x_minus - x_plus
        x_per = (x - x_plus) % interval + x_plus
        return sol_u(x_per)
    return u_per


# Save function and dictionaries
def save_func_dic(func, path):
    with open(path, "wb") as f:
        pickle.dump(func, f)


# Save a file
def save_file(w, path):
    path = path + ".npy.gz"
    with gzip.open(path, 'wb') as f:
        np.save(f, w)


# Creates a folder
def create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

def get_s_max():
    rho_s_zero = root(lambda rho: U_prime(rho) + h_prime(rho), 0.2*rho_max).x[0]
    v_s_zero = 1/rho_s_zero
    m_max = -h_bar_prime(v_s_zero)
    s_max = U_bar(v_s_zero) - m_max * v_s_zero
    return s_max

def get_m(v_s):
    m = -h_bar_prime(v_s)
    return m

def get_s(m, v_s):
    s = U_bar(v_s) - m*v_s
    return s

# Scientific notation formatter
def sci_notation(number:float, n_decimals: int=2) -> str:
    sci_notation = f"{number:e}"
    decimal_part, exponent = sci_notation.split("e")
    decimal_part = float(decimal_part)
    exponent = int(exponent)

    if decimal_part == 1.0:
        return r"1.00 $ \times 10^{" +str(exponent)+"}$"

    else:
        return fr"{decimal_part:.{n_decimals}f}"+r" $ \times 10^{" +str(exponent)+"}$"
