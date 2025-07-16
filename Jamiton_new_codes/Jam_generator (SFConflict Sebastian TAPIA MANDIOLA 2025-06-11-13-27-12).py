from Functions import *
from Utilities import *
from Config import *
from scipy.optimize import minimize_scalar

# Resuelve EDO con respecto a x
def ODE_jam_solve(tf, v_R, tau, m, s):
    # Resuelve EDO
    N_ODE = int(1e6)
    sol_v = solve_ivp(ode_jam_v, (0, tf), [v_R], t_eval=np.linspace(0, tf, N_ODE), args=[tau, m, s], dense_output=True)
    return sol_v


# Resuelve EDO con respecto a eta
def ODE_jam_solve_eta(eta_f, v_R, m, s):
    # Resuelve EDO
    N_ODE = int(1e6)
    sol_v = solve_ivp(ode_jam_v_eta, (0, eta_f), [v_R], t_eval=np.linspace(0, eta_f, N_ODE), args=[m, s], dense_output=True)
    return sol_v


# Genera jamitones
def jam_gen(v_s, tf, tau, v_minus=None, print_pars=True):

    global sol_rho
    global sol_u

    values_v = {}
    values_rho = {}
    values_u = {}

    # Parámetros del jamiton
    m = -h_bar_prime(v_s)
    s = U_bar(v_s) - m * v_s

    # Imprime parámetros
    if print_pars:
        print("Jamiton speed: ", round(s, 4))
        print("m = ", round(m, 4))

    # Existencia jamiton
    jam_exs = U_prime(v_to_rho(v_s)) + h_prime(v_to_rho(v_s)) < 0
    if not jam_exs:
        print("There is no jamiton. Try another rho_s value")
        pass

    # Jamiton maximal
    v_M = newton(lambda v: w_v(v, m, s), 40) #.x[0]
    v_R = newton(lambda v: r(v, m) - r(v_M, m), 10) #.x[0]

    # Jamiton actual
    if v_minus is None:
        print("Default v_minus...")
        v_minus = (v_M + v_s)/2
        #float(input("Escoja v_min (entre {vs} y {vM}): ".format(vs =round(v_s, 3), vM=round(v_M, 3))))
    v_plus = newton(lambda v: r(v, m) - r(v_minus, m), 8) #.x[0]

    # Guarda valores de v
    values_v["v_s"] = v_s
    values_v["v_M"] = v_M
    values_v["v_R"] = v_R
    values_v["v_minus"] = v_minus
    values_v["v_plus"] = v_plus

    # Guarda valores de rho
    values_rho["rho_s"] = v_to_rho(v_s)
    values_rho["rho_M"] = v_to_rho(v_M)
    values_rho["rho_R"] = v_to_rho(v_R)
    values_rho["rho_minus"] = v_to_rho(v_minus)
    values_rho["rho_plus"] = v_to_rho(v_plus)

    # Guarda valores de u
    values_u["u_s"] = rho_to_u(values_rho["rho_s"], m, s)
    values_u["u_M"] = rho_to_u(values_rho["rho_M"], m, s)
    values_u["u_R"] = rho_to_u(values_rho["rho_R"], m, s)
    values_u["u_minus"] = rho_to_u(values_rho["rho_minus"], m, s)
    values_u["u_plus"] = rho_to_u(values_rho["rho_plus"], m, s)

    # Solucion EDO
    sol_v = ODE_jam_solve(tf, v_R, tau, m, s)

    def sol_rho(x):
        return v_to_rho(sol_v.sol(x)[0])

    def sol_u(x):
        return rho_to_u(sol_rho(x), m, s)

    return values_v, values_rho, values_u, sol_v, sol_rho, sol_u, m, s


# Encuentra xs importantes
def find_xs(sol_v, values_v, rho_s, tf, x_init=None):

    # Rescata valores de v
    v_s = values_v["v_s"]
    v_minus = values_v["v_minus"]
    v_plus = values_v["v_plus"]


    #x_init = float(input("Ingrese x inicial para x_min: "))
    # Calcula cada x
    x_minus = minimize_scalar(lambda v: np.abs(sol_v.sol(v)[0] - v_minus)**2, bounds=(0, tf)).x
    x_plus = root(lambda v: sol_v.sol(v)[0] - v_plus, 0).x[0]
    #x_s = minimize_scalar(lambda v: np.abs(sol_v.sol(v)[0] - v_s)**2, bounds=(0, tf)).x #.x[0]

    x_to_plot = sol_v.t
    x_to_per = np.linspace(x_minus, sol_v.t[-1], 500)
    return x_to_plot, x_minus, x_plus, x_to_per


def jam_values(tau, rho_s, tf, x_init=None, v_minus=None, print_pars=True):
    global sol_rho_eta
    global sol_u_eta
    #v_f = 100
    #tf = 6000

    # Elección valores sónicos
    #rho_s = float(input("Ingrese rho_s: "))
    #t_f = float(input("Ingrese tiempo final de integración: "))
    rho_s *= rho_max
    v_s = 1/rho_s # Se necesita rho_s normalizado

    # Genera jamitones
    values_v, values_rho, values_u, sol_v, sol_rho, sol_u, m, s = jam_gen(v_s, tf, tau, v_minus=v_minus, print_pars=print_pars)

    # Resuelve
    sol_v_eta = ODE_jam_solve_eta(tf, values_v["v_R"], m, s)
    def sol_rho_eta(eta):
        return v_to_rho(sol_v_eta.sol(eta)[0])
    def sol_u_eta(eta):
        return rho_to_u(sol_rho_eta(eta), m, s)

    # Rescata x's
    xs = find_xs(sol_v, values_v, rho_s, tf, x_init)
    x_minus = xs[1]
    x_plus = xs[2]

    # Arreglo con jamiton
    x_jam = np.linspace(x_plus, x_minus, 100)
    #print(sol_rho(x_plus)/rho_max, sol_rho(x_minus)/rho_max)
    if print_pars:
        print("x_+: ", round(x_plus, 4), "x_-: ", round(x_minus, 4))

    return x_minus, x_plus, sol_v, sol_v_eta, m, s, values_v
