from Utilities import *
from Config import *

# Fluxes
def flux_rho(rho, y):
    return y-rho*h(rho)

def flux_y(rho, y):
    return (y**2)/rho - h(rho)*y


# Lax - Friedrichs
def F_LxF_rho(rho, y, l):
    rho_left = np.roll(rho, 1)
    y_left = np.roll(y, 1)

    rho_right = np.roll(rho, -1)
    y_right = np.roll(y, -1)

    flux = flux_rho(rho, y)
    flux_right = flux_rho(rho_right, y_right)
    flux_left = flux_rho(rho_left, y_left)

    flux_bar_right = (flux_right + flux)/2 - (rho_right - rho)/(2*l)
    flux_bar_left = (flux_left + flux)/2 - (rho - rho_left)/(2*l)

    return flux_bar_right - flux_bar_left


def F_LxF_y(rho, y, l):
    rho_left = np.roll(rho, 1)
    y_left = np.roll(y, 1)

    rho_right = np.roll(rho, -1)
    y_right = np.roll(y, -1)

    flux = flux_y(rho, y)
    flux_right = flux_y(rho_right, y_right)
    flux_left = flux_y(rho_left, y_left)

    flux_bar_right = (flux_right + flux)/2 - (y_right - y)/(2*l)
    flux_bar_left = (flux_left + flux)/2 - (y - y_left)/(2*l)

    return flux_bar_right - flux_bar_left


# HLL solver
# Flujo con HLL
def F_HLL_rho(rho, y, l):

    rho_left =  np.roll(rho, 1)
    y_left = np.roll(y, 1)

    rho_right = np.roll(rho, -1)
    y_right = np.roll(y, -1)

    F_rho = flux_HLL_rho(rho, y, rho_right, y_right) - flux_HLL_rho(rho_left, y_left, rho, y)

    return F_rho

def F_HLL_y(rho, y, l):

    rho_left =  np.roll(rho, 1)
    y_left = np.roll(y, 1)

    rho_right = np.roll(rho, -1)
    y_right = np.roll(y, -1)

    F_y = flux_HLL_y(rho, y, rho_right, y_right) - flux_HLL_y(rho_left, y_left, rho, y)

    return F_y


# Solver HLL para riemann
def flux_HLL_rho(rho_l, y_l, rho_r, y_r):

    # Obtiene velocidades
    u_l = u_y(rho_l, y_l)
    u_r = u_y(rho_r, y_r)

    l_1_l = u_r - rho_r * h_prime(rho_r)
    l_1_r = u_l - rho_l * h_prime(rho_l)
    l_2_l = u_l
    l_2_r = u_r

    s_l = np.min([l_1_l, l_1_r])
    s_r = np.max([l_2_l, l_2_r])

    s_R_plus = np.max([s_r, 0])
    s_l_minus = np.min([s_l, 0])

    F_l = flux_rho(rho_l, y_l)
    F_r = flux_rho(rho_r, y_r)

    F_hat = (s_R_plus * F_l - s_l_minus * F_r + s_l_minus * s_R_plus * (rho_r - rho_l))/(s_R_plus - s_l_minus)
    return F_hat

def flux_HLL_y(rho_l, y_l, rho_r, y_r):

    # Obtiene velocidades
    u_l = u_y(rho_l, y_l)
    u_r = u_y(rho_r, y_r)

    l_1_l = u_r - rho_r * h_prime(rho_r)
    l_1_r = u_l - rho_l * h_prime(rho_l)
    l_2_l = u_l
    l_2_r = u_r

    s_l = np.min([l_1_l, l_1_r])
    s_r = np.max([l_2_l, l_2_r])

    s_R_plus = np.max([s_r, 0])
    s_l_minus = np.min([s_l, 0])

    F_l = flux_y(rho_l, y_l)
    F_r = flux_y(rho_r, y_r)

    F_hat = (s_R_plus * F_l - s_l_minus * F_r + s_l_minus * s_R_plus * (y_r - y_l))/(s_R_plus - s_l_minus)
    return F_hat


# Simulation function
def solve_ARZ(Q_0, x, tau, T, Nt, F_rho, F_y, s_max, as_array=True, N_elem=None, print_step=True):

    t = np.linspace(0, T, Nt)
    dt = t[1] - t[0]

    # salto espacial
    dx = x[1] - x[0]
    N = len(x)

    CFL = 2 * s_max * dt/dx < 1

    if not CFL:
        print("CFL is not satisfied")
        pass

    # Init condition
    if as_array:
        rho = np.zeros((Nt, N))
        y = np.zeros((Nt, N))

        rho[0] = Q_0[0]
        y[0] = Q_0[1]

    elif N_elem is not None:
        N_times = int(Nt/N_elem)+1

        t_save = np.zeros(N_elem+1)
        rho_save = np.zeros((N_elem+1, N))
        y_save = np.zeros((N_elem+1, N))

        rho_save[0] = Q_0[0]
        y_save[0] = Q_0[1]
        save_index = 0

        rho = Q_0[0]
        y = Q_0[1]

    else:
        rho = Q_0[0]
        y = Q_0[1]

    l = dt/dx
    alpha = dt/tau

    # Solving
    for n in range(Nt-1):
        if print_step:
            print("Step ", n, " of ", Nt)

        if as_array:
            rho_n = rho[n]
            y_n = y[n]

        else:
            rho_n = rho
            y_n = y
        
        # Godunov step
        rho_sig = rho_n - l * F_rho(rho_n, y_n, l)
        y_sig = y_n - l * F_y(rho_n, y_n, l)
        
        # Relaxation term (implicit)
        y_sig_ = (alpha/(1+alpha)) * rho_sig * (U(rho_sig) + h(rho_sig)) + (1/(1+alpha)) * y_sig
        y_sig = y_sig_

        # step update
        if as_array:
            rho[n+1] = rho_sig
            y[n+1] = y_sig

        else:
            rho = rho_sig
            y = y_sig

            if N_elem is not None and (n+1)%N_times == 0:
                print("Saved time: ", t[n+1], "n = ", n+1)
                rho_save[save_index] = rho
                y_save[save_index] = y
                t_save[save_index] = t[n+1]
                save_index += 1

    if N_elem is not None:
        rho_save[-1] = rho
        y_save[-1] = y
        t_save[-1] = t[-1]
        return t_save, rho_save, y_save

    return t, rho, y
