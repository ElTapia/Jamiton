from Utilities import *
from Config import *
from Jam_generator import *


def Q_0_collide(dx, tau, rho_s_1, rho_s_2, v_minus, tf_1, tf_2, x_init=None, print_pars=True):
    # Primer jamiton
    x_minus_1, x_plus_1, sol_v_1, sol_v_eta_1, m_1, s_1, values_v_1 = jam_values(tau, rho_s_1, tf_1, v_minus=v_minus, print_pars=print_pars)

    sol_rho_1 = lambda x: 1/sol_v_1.sol(x)[0]
    sol_u_1 = lambda x: rho_to_u(sol_rho_1(x), m_1, s_1)

    # Segundo jamiton
    x_minus_2, x_plus_2, sol_v_2, sol_v_eta_2, m_2, s_2, values_v_2 = jam_values(tau, rho_s_2, tf_2, v_minus=v_minus, print_pars=print_pars)

    sol_rho_2 = lambda x: 1/sol_v_2.sol(x)[0]
    sol_u_2 = lambda x: rho_to_u(sol_rho_2(x), m_2, s_2)

    # Jamitones compatibles
    v_minus_1 = values_v_1["v_minus"]
    v_minus_2 = values_v_2["v_minus"]

    compatible = np.isclose(v_minus_1, v_minus_2, atol=1e-2)

    if not compatible:
        print(v_minus_1, v_minus_2)
        print("Incompatible Jamitons")
        return None

    L = x_minus_2 + (x_minus_1 - x_plus_2) - x_plus_1
    N = int(L/dx + 1)
    x_to_plot = np.linspace(0, x_minus_2+(x_minus_1 - x_plus_2) - x_plus_1, N)

    def rho_sol_combined(x):
        if x_plus_1 <= x and x <= x_minus_1:
            return sol_rho_1(x)
    
        elif x_plus_2 <= x - (x_minus_1 - x_plus_2) and x - (x_minus_1 - x_plus_2) <= x_minus_2:
            return sol_rho_2(x-(x_minus_1 - x_plus_2))

    def u_sol_combined(x):
        if x_plus_1 <= x and x <= x_minus_1:
            return sol_u_1(x)
    
        elif x_plus_2 <= x - (x_minus_1 - x_plus_2) and x - (x_minus_1 - x_plus_2) <= x_minus_2:
            return sol_u_2(x-(x_minus_1 - x_plus_2))

    rho_sol_combined = np.vectorize(rho_sol_combined)
    u_sol_combined = np.vectorize(u_sol_combined)

    def rho_per(x):
        interval = x_minus_2+(x_minus_1 - x_plus_2) - x_plus_1
        x_per = (x - x_plus_1) % interval + x_plus_1
        return rho_sol_combined(x_per)

    def u_per(x):
        interval = x_minus_2+(x_minus_1 - x_plus_2) - x_plus_1
        x_per = (x - x_plus_1) % interval + x_plus_1
        return u_sol_combined(x_per)

    rho_0 = rho_per(x_to_plot)
    u_0 = u_per(x_to_plot)
    y_0 = rho_0 * (u_0 + h(rho_0))

    Q_0_ = np.zeros([2, len(x_to_plot)])
    Q_0_[0] = rho_0
    Q_0_[1] = y_0
    
    return Q_0_, x_to_plot

def Q_0_collide_fix(dx, tau, jam_1_data, rho_s_2, v_minus, tf_2, x_init=None, print_pars=True):

    # First Jamiton
    x_minus_1 = jam_1_data["x_minus"]
    x_plus_1 = jam_1_data["x_plus"]
    sol_rho_1 = jam_1_data["sol_rho"]
    sol_u_1 = jam_1_data["sol_u"]

    # Segundo jamiton
    x_minus_2, x_plus_2, sol_v_2, sol_v_eta_2, m_2, s_2, values_v_2 = jam_values(tau, rho_s_2, tf_2, v_minus=v_minus, print_pars=print_pars)

    sol_rho_2 = lambda x: 1/sol_v_2.sol(x)[0]
    sol_u_2 = lambda x: rho_to_u(sol_rho_2(x), m_2, s_2)

    L = x_minus_2 + (x_minus_1 - x_plus_2) - x_plus_1
    N = int(L/dx + 1)
    x_to_plot = np.linspace(0, x_minus_2+(x_minus_1 - x_plus_2) - x_plus_1, N)

    def rho_sol_combined(x):
        if x_plus_1 <= x and x <= x_minus_1:
            return sol_rho_1(x)
    
        elif x_plus_2 <= x - (x_minus_1 - x_plus_2) and x - (x_minus_1 - x_plus_2) <= x_minus_2:
            return sol_rho_2(x-(x_minus_1 - x_plus_2))

    def u_sol_combined(x):
        if x_plus_1 <= x and x <= x_minus_1:
            return sol_u_1(x)
    
        elif x_plus_2 <= x - (x_minus_1 - x_plus_2) and x - (x_minus_1 - x_plus_2) <= x_minus_2:
            return sol_u_2(x-(x_minus_1 - x_plus_2))

    rho_sol_combined = np.vectorize(rho_sol_combined)
    u_sol_combined = np.vectorize(u_sol_combined)

    def rho_per(x):
        interval = x_minus_2+(x_minus_1 - x_plus_2) - x_plus_1
        x_per = (x - x_plus_1) % interval + x_plus_1
        return rho_sol_combined(x_per)

    def u_per(x):
        interval = x_minus_2+(x_minus_1 - x_plus_2) - x_plus_1
        x_per = (x - x_plus_1) % interval + x_plus_1
        return u_sol_combined(x_per)

    rho_0 = rho_per(x_to_plot)
    u_0 = u_per(x_to_plot)
    y_0 = rho_0 * (u_0 + h(rho_0))

    Q_0_ = np.zeros([2, len(x_to_plot)])
    Q_0_[0] = rho_0
    Q_0_[1] = y_0
    
    return Q_0_, x_to_plot

def get_rhos_violate():
    rho_min_scc = root(lambda rho: U_prime(rho) + h_prime(rho), 0.1*rho_max, method="lm").x[0]
    rho_max_scc = root(lambda rho: U_prime(rho) + h_prime(rho), 0.7*rho_max, method="lm").x[0]

    rhos_violate = np.linspace(rho_min_scc, rho_max_scc, 1_500)
    return rhos_violate


def get_best_rho_s(rhos_violate):

    L = -np.inf
    for i in range(len(rhos_violate)):
        rho_scc = rhos_violate[i]
        v_scc = 1/rho_scc
        m_scc = get_m(v_scc)
        s_scc = get_s(m_scc, v_scc)
        w_scc = lambda v: w_v(v, m_scc, s_scc)
        v_M = root(w_scc, 100, method="hybr")
        L_sig = v_M.x[0] - v_scc

        if L_sig < L:
            v_medio = (v_scc + v_M.x[0])/2
            return rho_scc, i, v_medio

        L = L_sig


def get_rho_s_tests():

    rhos_violate = get_rhos_violate()

    rho_s_test, i_test, v_medio_test = get_best_rho_s(rhos_violate)
    rhos_candidatos = np.delete(rhos_violate, i_test)
    rhos_to_test = []
    
    for j in range(len(rhos_candidatos)):
        rho_cand = rhos_candidatos[j]
        v_cand = 1/rho_cand
        m_cand = get_m(v_cand)
        s_cand = get_s(m_cand, v_cand)
        w_cand = lambda v: w_v(v, m_cand, s_cand)
        v_M_cand = root(w_cand, 100, method="hybr")
        
        if v_cand < v_medio_test and v_medio_test < v_M_cand.x[0]:
            rhos_to_test += [rho_cand]
    
    rhos_to_test = np.array(rhos_to_test)
    return rhos_to_test, rho_s_test, v_medio_test
