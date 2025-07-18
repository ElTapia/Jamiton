import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root
from scipy.interpolate import interp1d
from scipy.integrate import solve_bvp, solve_ivp
from adjustText import adjust_text
from functions_new import *

# Estilo de gráficos
plt.style.use('bmh')

# Gráfico de w
def plot_w(values_v, m, s, v_f):

    # Toma valores de v
    v_s = values_v["v_s"]
    v_M = values_v["v_M"]

    texts = []
    v_to_plot = np.linspace(8, v_f, 1_000)
    plt.plot(v_to_plot, w_v(v_to_plot, m, s), label="w(v)", zorder=0, color="black", lw=1)

    plt.hlines(0, 1, v_f, ls="--", label="Cero", zorder=1)
    plt.scatter(v_s, w_v(v_s, m, s), color="red", zorder=2)
    texts += [plt.annotate(r"$v_s$", (v_s, w_v(v_s, m, s)), fontsize=20)]

    # jamiton maximal
    plt.scatter(v_M, w_v(v_M, m, s), color="green", zorder=2)
    texts += [plt.annotate(r"$v_M$", (v_M, w_v(v_M, m, s)), fontsize=20)]

    plt.xlabel(r"$v$", fontsize=20)
    plt.ylabel(r"$w(v)$", fontsize=20)
    plt.legend(fontsize=12)

    plt.xlim(v_s/1.5, 1.4*v_M)
    plt.ylim(w_v(v_s, m, s)-1.5, np.max(w_v(v_to_plot, m, s))+0.5)
    adjust_text(texts, only_move={'points':'y', 'texts':'y'})
    plt.tick_params(left = False , labelleft = False , 
                    labelbottom = False, bottom = False) 

    # TODO: Agregar savefig en charts
    # plt.savefig()
    plt.show()

    return v_M


# Gráfico de r
def plot_r(values_v, v_f, m, s):

    # Rescata valores de v
    v_s = values_v["v_s"]
    v_M = values_v["v_M"]
    v_R = values_v["v_R"]
    v_minus = values_v["v_minus"]
    v_plus = values_v["v_plus"]

    # Inicia gráfico
    v_to_plot = np.linspace(8, v_f, 1000)#15, 50, 1000)#
    texts = []
    plt.plot(v_to_plot, r(v_to_plot, m), zorder=0, color="black", label="r(v)", lw = 0.8)

    # Valores r
    r_min = r(v_s, m)
    r_max = r(v_M, m)
    r_R = r(v_R, m)
    r_minus = r(v_minus, m)
    r_plus = r(v_plus, m)

    # Velocidad sonica
    plt.scatter(v_s, r_min, color="red", zorder=2)
    texts += [plt.annotate(r"$v_s$", (v_s, r_min), fontsize=20)]

    # jamiton maximal
    plt.scatter(v_M, r_max, color="green", zorder=2)
    texts += [plt.annotate(r"$v_M$", (v_M, r_max), fontsize=20)]

    plt.scatter(v_R, r(v_R, m), color="green", zorder=2)
    texts += [plt.annotate(r"$v_R$", (v_R, r_R), fontsize=20)]

    # Linea de jamiton maximal
    plt.hlines(r_max, v_R, v_M, color="blue", ls="--", label="jamiton maximal", zorder=1)

    # Jamiton actual
    plt.scatter(v_minus, r_minus, color="purple", zorder=2)
    texts += [plt.annotate(r"$v_-$", (v_minus, r_minus), fontsize=20)]

    plt.scatter(v_plus, r_plus, color="purple", zorder=2)
    texts += [plt.annotate(r"$v_+$", (v_plus, r_plus), fontsize=20)]

    # Linea de jamiton actual
    plt.hlines(r_minus, v_plus, v_minus, color="brown", ls="--", label="jamiton actual", zorder=1)

    plt.xlabel(r"$v$", fontsize=20)
    plt.ylabel(r"$r(v)$", fontsize=20)

    plt.xlim(v_R/1.2, v_M*1.1)
    plt.ylim(r_min/1.05, r_R*1.05)

    adjust_text(texts)

    plt.tick_params(left = False , labelleft = False , 
                    labelbottom = False, bottom = False) 
    plt.legend(fontsize=12)

    #plt.savefig("Jamitones/r_rho_{}.png".format(rho_s/rho_max))
    plt.show()


# Gráfico diagrama fundamental
def plot_Q(values_v, m, s):
    texts = []

    # Vector con densidades
    rhos = np.linspace(0, rhomax-1e-5, 1_000)
    # Diagrama fundamental
    plt.plot(rhos/rhomax, Q_e(rhos), zorder=0, color="black", label="Curva de equilibrio", lw=1)

    # Valores de v
    v_s = values_v["v_s"]
    v_M = values_v["v_M"]
    v_R = values_v["v_R"]
    v_minus = values_v["v_minus"]
    v_plus = values_v["v_plus"]

    # Valores de rho
    rho_plus = 1/v_plus
    rho_minus = 1/v_minus
    rho_M = 1/v_M
    rho_R = 1/v_R
    rho_s = 1/v_s

    plt.scatter(rho_s/rhomax, Q_e(rho_s), color="red", zorder=3)
    texts += [plt.annotate(r"$\rho_s$", (rho_s/rhomax, Q_e(rho_s)), fontsize=15)]

    plt.scatter(rho_plus/rhomax, m + s * rho_plus, color="purple", zorder=3)
    texts += [plt.annotate(r"$\rho_+$", (rho_plus/rhomax, m + s * rho_plus), fontsize=15)]
    plt.scatter(rho_minus/rhomax, m + s * rho_minus, color="purple", zorder=3)
    texts += [plt.annotate(r"$\rho_-$", (rho_minus/rhomax, m + s * rho_minus), fontsize=15)]

    plt.scatter(rho_M/rhomax, m + s * rho_M, color="green", zorder=3)
    texts += [plt.annotate(r"$\rho_M$", (rho_M/rhomax, m + s * rho_M), fontsize=15)]
    plt.scatter(rho_R/rhomax,  m + s * rho_R, color="green", zorder=3)
    texts += [plt.annotate(r"$\rho_R$", (rho_R/rhomax,  m + s * rho_R), fontsize=15)]

    plt.plot([rho_plus/rhomax, rho_minus/rhomax], [m + s * rho_plus, m + s * rho_minus], color="brown", ls="-", 
            zorder=2, label="Jamiton actual")
    plt.plot([rho_M/rhomax, rho_R/rhomax], [m + s * rho_M, m + s * rho_R], color="blue", ls="-", 
            zorder=1, label="Jamiton maximal")

    plt.ylabel(r"$Q(\rho)$", fontsize=20)
    plt.xlabel(r"$\rho$", fontsize=20)
    plt.legend(fontsize=12)
    adjust_text(texts)

    #plt.savefig("Jamitones/Q_rho_{}.png".format(rho_s/rho_max))
    plt.show()


# Gráfico de v
def plot_v(sol_v, values_v, t_f, xs):

    # Rescata x_s importantes
    x_to_plot, x_minus, x_plus, x_s, x_to_per = xs

    # Rescata valores de v
    v_s = values_v["v_s"]
    v_M = values_v["v_M"]
    v_R = values_v["v_R"]
    v_minus = values_v["v_minus"]
    v_plus = values_v["v_plus"]

    # x y v de jamiton actual
    x_jam = np.linspace(x_plus, x_minus, 500)
    v_jam = sol_v.sol(x_jam)[0]

    # Cadena de jamitones
    def v_per(x):
        interval = x_plus - x_minus
        return sol_v.sol((x - x_minus) % interval + x_minus)[0]

    # Jamiton actual y cadena
    texts = []
    plt.plot(x_jam, v_jam, zorder=1, label="Jamiton actual", color="brown")
    plt.plot(x_to_per, v_per(x_to_per), zorder=0, color="brown", ls="--", label="Cadena de jamitones")

    # Límites jamiton
    plt.scatter(x_plus, v_plus, zorder=2, color="green")
    texts += [plt.annotate(r"$v_+$", (x_plus, v_plus), fontsize=20)]
    plt.scatter(x_minus, v_minus, zorder=2, color="green")
    texts += [plt.annotate(r"$v_-$", (x_minus, v_minus), fontsize=20)]
    plt.scatter(x_s, v_s, zorder=2, color="red", label="Punto sónico")
    texts += [plt.annotate(r"$v_s$", (x_s, v_s), fontsize=20)]

    # Jamiton maximal
    plt.plot(x_to_plot, sol_v.y[0], zorder=0, label="Jamiton maximal")
    plt.scatter(0, v_R, color="purple")
    texts += [plt.annotate(r"$v_R$", (0, v_R), fontsize=20)]
    plt.hlines(v_M, 0, t_f, color="purple", zorder=0, ls="--")
    texts += [plt.annotate(r"$v_M$", (t_f, v_M), fontsize=20)]

    # Ajusta gráfico
    plt.xlabel(r"$x$", fontsize=20)
    plt.ylabel(r"$v(x)$", fontsize=20)
    plt.legend(fontsize=12)
    adjust_text(texts)

    #plt.savefig("Jamitones/v_rho_{}.png".format(rho_s/rho_max))
    plt.show()


def plot_rho(sol_rho, values_rho, t_f, xs):
    # Obtiene x's
    x_to_plot, x_minus, x_plus, x_s, x_to_per = xs

    # Rescata valores de rho
    rho_s = values_rho["rho_s"]
    rho_M = values_rho["rho_M"]
    rho_R = values_rho["rho_R"]
    rho_minus = values_rho["rho_minus"]
    rho_plus = values_rho["rho_plus"]

    # x y rho actual
    x_jam = np.linspace(x_plus, x_minus, 500)
    rho_jam = sol_rho(x_jam)
    rho_y = sol_rho(x_to_plot)

    # Cadena de jamitones
    def rho_per(x):
        interval = x_plus - x_minus
        return sol_rho((x - x_minus) % interval + x_minus)

    # Jamiton actual y cadena
    texts = []
    plt.plot(x_jam, rho_jam/rhomax, zorder=1, label="Jamiton actual", color="brown")
    plt.plot(x_to_per, rho_per(x_to_per)/rhomax, zorder=0, color="brown", ls="--", label="Cadena de jamitones")

    # Límites jamiton
    plt.scatter(x_plus, rho_plus/rhomax, zorder=2, color="green")
    texts += [plt.annotate(r"$\rho_+$", (x_plus, rho_plus/rhomax), fontsize=20)]
    plt.scatter(x_minus, rho_minus/rhomax, zorder=2, color="green")
    texts += [plt.annotate(r"$\rho_-$", (x_minus, rho_minus/rhomax), fontsize=20)]
    plt.scatter(x_s, rho_s/rhomax, zorder=2, color="red", label="Punto sónico")
    texts += [plt.annotate(r"$\rho_s$", (x_s, rho_s/rhomax), fontsize=20)]

    # Jamiton maximal
    plt.plot(x_to_plot, rho_y/rhomax, zorder=0, label="Jamiton maximal")
    plt.scatter(0, rho_R/rhomax, color="purple")
    texts += [plt.annotate(r"$\rho_R$", (0, rho_R/rhomax), fontsize=20)]
    plt.hlines(rho_M/rhomax, 0, t_f, color="purple", zorder=0, ls="--")
    texts += [plt.annotate(r"$\rho_M$", (t_f, rho_M/rhomax), fontsize=20)]

    plt.xlabel(r"$x$", fontsize=20)
    plt.ylabel(r"$\rho(x)$", fontsize=20)
    plt.legend(fontsize=12)
    adjust_text(texts)

    #plt.savefig("Jamitones/rho_rho_{}.png".format(rho_s/rho_max))
    plt.show()


def plot_u(sol_u, values_u, t_f, xs):
    # Obtiene x's
    x_to_plot, x_minus, x_plus, x_s, x_to_per = xs

    # Rescata valores de u
    u_s = values_u["u_s"]
    u_M = values_u["u_M"]
    u_R = values_u["u_R"]
    u_minus = values_u["u_minus"]
    u_plus = values_u["u_plus"]

    # x y u jamiton actual
    x_jam = np.linspace(x_plus, x_minus, 500)
    u_jam = sol_u(x_jam)
    u_y = sol_u(x_to_plot)

    # Cadena de jamitones
    def u_per(x):
        interval = x_plus - x_minus
        return sol_u((x - x_minus) % interval + x_minus)

    # Jamiton actual
    texts = []
    plt.plot(x_jam, u_jam, zorder=1, label="Jamiton actual", color="brown")
    plt.plot(x_to_per, u_per(x_to_per), zorder=0, color="brown", ls="--", label="Cadena de jamitones")

    # Límites jamiton
    plt.scatter(x_plus, u_plus, zorder=2, color="green")
    texts += [plt.annotate(r"$u_+$", (x_plus, u_plus), fontsize=20)]
    plt.scatter(x_minus, u_minus, zorder=2, color="green")
    texts += [plt.annotate(r"$u_{-}$", (x_minus, u_minus), fontsize=20)]
    plt.scatter(x_s, u_s, zorder=2, color="red", label="Punto sónico")
    texts += [plt.annotate(r"$u_s$", (x_s, u_s), fontsize=20)]

    # Jamiton maximal
    plt.plot(x_to_plot, u_y, zorder=0, label="Jamiton maximal")
    plt.scatter(0, u_R, color="purple")
    texts += [plt.annotate(r"$u_R$", (0, u_R), fontsize=20)]
    plt.hlines(u_M, 0, t_f, color="purple", zorder=0, ls="--")
    texts += [plt.annotate(r"$u_M$", (t_f, u_M), fontsize=20)]

    # Ajusta gráfico
    plt.xlabel(r"$x$", fontsize=20)
    plt.ylabel(r"$u(x)$", fontsize=20)
    plt.legend(fontsize=12)
    adjust_text(texts)

    #plt.savefig("Jamitones/u_rho_{}.png".format(rho_s/rho_max))
    plt.show()


# Resuelve EDO con respecto a x
def ODE_jam_solve(t_f, v_R, tau, m, s):
    # Resuelve EDO
    sol_v = solve_ivp(ode_jam_v, (0, t_f), [v_R], t_eval=np.linspace(0, t_f, 10_000), args=[tau, m, s], dense_output=True)
    return sol_v

# Resuelve EDO con respecto a eta
def ODE_jam_solve_eta(eta_f, v_R, m, s):
    # Resuelve EDO
    sol_v = solve_ivp(ode_jam_v_eta, (0, eta_f), [v_R], t_eval=np.linspace(0, eta_f, 10_000), args=[m, s], dense_output=True)
    return sol_v


# Encuentra xs importantes
def find_xs(sol_v, values_v, rho_s, x_init=None):

    # Rescata valores de v
    v_s = values_v["v_s"]
    v_minus = values_v["v_minus"]
    v_plus = values_v["v_plus"]

    if x_init is None:
        if 0.24 < rho_s < 0.4:
            x_init = 0

        if 0.4 <= rho_s < 0.6:
            x_init = 90

        else:
            x_init = 250
    #x_init = float(input("Ingrese x inicial para x_min: "))
    # Calcula cada x
    x_minus = root(lambda v: sol_v.sol(v)[0] - v_minus, 10).x[0]
    x_plus = root(lambda v: sol_v.sol(v)[0] - v_plus, 0).x[0]
    x_s = root(lambda v: sol_v.sol(v)[0] - v_s, 0).x[0]
    x_to_plot = sol_v.t
    x_to_per = np.linspace(x_minus, sol_v.t[-1], 500)

    return x_to_plot, x_minus, x_plus, x_s, x_to_per


# Genera jamitones
def jam_gen(v_s, t_f, tau):

    values_v = {}
    values_rho = {}
    values_u = {}

    # Parámetros del jamiton
    m = -h_bar_prime(v_s)
    s = U_bar(v_s) - m * v_s

    # Imprime parámetros
    print("Velocidad jamiton: ", s)
    print("m= ", m)

    # Existencia jamiton
    jam_exs = U_prime(v_to_rho(v_s)) + h_prime(v_to_rho(v_s)) < 0
    if not jam_exs:
        print("No existe jamiton, pruebe otro valor de rho_s")
        pass

    # Jamiton maximal
    v_M = root(lambda v: w_v(v, m, s), 40).x[0]
    v_R = root(lambda v: r(v, m) - r(v_M, m), 10).x[0]

    # Jamiton actual
    v_minus = float(input("Escoja v_min (entre {vs} y {vM}): ".format(vs =round(v_s, 3), vM=round(v_M, 3))))
    v_plus = root(lambda v: r(v, m) - r(v_minus, m), 8).x[0]

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
    sol_v = ODE_jam_solve(t_f, v_R, tau, m, s)
    sol_rho = lambda x: v_to_rho(sol_v.sol(x)[0])
    sol_u = lambda x: rho_to_u(sol_rho(x), m, s)

    return values_v, values_rho, values_u, sol_v, sol_rho, sol_u, m, s

# Transforma v a rho
def v_to_rho(v):
    rho = 1/v
    return rho

# Transforma rho a u
def rho_to_u(rho, m, s):
    u = (m/rho) + s
    return u

# Inicia programa
def init_program(tau, rho_s, plotear=False, x_init=None):
    v_f = 100
    t_f = 6000

    # Elección valores sónicos
    #rho_s = float(input("Ingrese rho_s: "))
    #t_f = float(input("Ingrese tiempo final de integración: "))
    rho_s *= rhomax
    v_s = 1/rho_s # Se necesita rho_s normalizado

    # Genera jamitones
    values_v, values_rho, values_u, sol_v, sol_rho, sol_u, m, s = jam_gen(v_s, t_f, tau)

    # Resuelve para eta
    sol_v_eta = ODE_jam_solve_eta(t_f, values_v["v_R"], m, s)
    sol_rho_eta = lambda eta: v_to_rho(sol_v_eta.sol(eta)[0])
    sol_u_eta = lambda eta: rho_to_u(sol_rho_eta(eta), m, s)

    # Rescata x's
    xs = find_xs(sol_v, values_v, rho_s, x_init)
    x_minus = xs[1]
    x_plus = xs[2]

    # Arreglo con jamiton
    x_jam = np.linspace(x_plus, x_minus, 100)
    #print(sol_rho(x_plus)/rhomax, sol_rho(x_minus)/rhomax)
    print(x_plus, x_minus)

    # Plotea
    # plotear = input("¿Desea graficar? (y/n): ")
    if plotear:  # == "y":
        plot_w(values_v, m, s, v_f)
        plot_r(values_v, v_f, m, s)
        plot_Q(values_v, m, s)
        plot_v(sol_v, values_v, t_f, xs)
        plot_rho(sol_rho, values_rho, t_f, xs)
        plot_u(sol_u, values_u, t_f, xs)

    return x_minus, x_plus, sol_rho, sol_u, sol_rho_eta, sol_u_eta, s
