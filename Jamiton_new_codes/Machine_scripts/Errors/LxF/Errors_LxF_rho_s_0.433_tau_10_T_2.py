import sys
sys.path.append("../../../")

from Config import *
from Utilities import *
from Solver import *
from Jam_generator import *
from Functions import *

s_max = get_s_max()

rho_s = 0.433
v_minus = 26
tau = 10
dxs = [(2)**(-j)/10 for j in range(8)]
T = 2

base_folder = "rho_s_{rho}_Final_time_{T}_local".format(rho=rho_s, T=T)
create_folder(base_folder)
base_path = base_folder + "/"

print("Solving for rho_s ", rho_s)


tau_folder = base_path + "tau_{}".format(tau)
create_folder(tau_folder)
tau_path = tau_folder + "/"

#rhos = np.zeros(len(dxs))
#us = np.zeros(len(dxs))

print("Creating Jamiton rho_s = ", round(rho_s, 4), " ...")

x_minus, x_plus, sol_v, sol_v_eta, m, s, values_v = jam_values(tau, rho_s, 150, v_minus=v_minus)

sol_v_filename = tau_path + "sol_v.pck"
sol_v_eta_filename = tau_path + "sol_v_eta.pck"

#save_func_dic(sol_v, sol_v_filename)
#save_func_dic(sol_v_eta, sol_v_eta_filename)

sol_rho = lambda x: 1/sol_v.sol(x)[0]
sol_u = lambda x: rho_to_u(sol_rho(x), m, s)

sol_rho_per = rho_per_gen(sol_rho, x_plus, x_minus)
sol_u_per = u_per_gen(sol_u, x_plus, x_minus)

L = np.abs(x_minus-x_plus)

for n in range(len(dxs)):
    dx = dxs[n]
    dt = dx/(2.1*s_max)

    print("Solving for dx ", dx)
    N = int(L/dx + 1)
    Nt = int(T/dt + 1)

    x = np.linspace(0, L, N)

    rho_0 = sol_rho_per(x)
    u_0 = sol_u_per(x)
    y_0 = rho_0 * (u_0 + h(rho_0))
    Q_0 = [rho_0, y_0]

    t, rho, y = solve_ARZ(Q_0, x, tau, T, Nt, F_LxF_rho, F_LxF_y, s_max, as_array=False)
    u = y/rho - h(rho)

    save_file(rho, tau_path+"rho_dx_{}".format(dx))
    save_file(u, tau_path+"u_dx_{}".format(dx))

print("Done!")


