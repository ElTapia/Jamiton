import sys
sys.path.append("../../../")

from Config import *
from Utilities import *
from Solver import *
from Jam_generator import *
from Functions import *
from Collisions import *

tau = 1
j = 2

base_folder = "Colls_{j}_tau_{tau}".format(j=j, tau=tau)
create_folder(base_folder)
base_path = base_folder + "/"

rho_base_folder = base_path + "rho"
create_folder(rho_base_folder)
rho_base_path = rho_base_folder + "/"

y_base_folder = base_path + "y"
create_folder(y_base_folder)
y_base_path = y_base_folder + "/"

t_base_folder = base_path + "t"
create_folder(t_base_folder)
t_base_path = t_base_folder + "/"

rhos_to_test, rho_s_test, v_minus = get_rho_s_tests()
rho_s_test = rho_s_test/rho_max
tf_test = 20

total = int(len(rhos_to_test)/4)
rhos_to_collide = rhos_to_test[j*total:(j+1)*total]

# Test Jamiton
x_minus_t, x_plus_t, sol_v_t, sol_v_eta_t, m_t, s_t, values_v_t = jam_values(tau, rho_s_test, tf_test, v_minus=v_minus, print_pars=False)
sol_rho_t = lambda x: 1/sol_v_t.sol(x)[0]
sol_u_t = lambda x: rho_to_u(sol_rho_t(x), m_t, s_t)

jam_test_data = {"x_minus": x_minus_t, "x_plus": x_plus_t, "sol_rho":sol_rho_t, "sol_u":sol_u_t}

# Discretization
s_max = get_s_max()
dx = 2**(-4)
dt = dx/(2.1*s_max)

for j in range(len(rhos_to_collide)):
    rho_s = rhos_to_collide[j]/rho_max
    print("Collision ", j, " of ", len(rhos_to_collide))

    if rho_s < 0.512:
        tf = 50

    elif 0.512 < rho_s and rho_s < 0.559:
        tf = 150

    else:
        tf = 300

    Q_col, x = Q_0_collide_fix(dx, tau, jam_test_data, rho_s, v_minus, tf, print_pars=False)
    T = 30
    if np.fabs(rho_s - rho_s_test)< 0.04:
        T = 60

    # Simulation
    Nt = int(T/dt) + 1

    t, rho, y = solve_ARZ(Q_col, x, tau, T, Nt, F_HLL_rho, F_HLL_y, s_max, as_array=False, print_step=False)

    save_file(rho, rho_base_path+"rho_coll_{}".format(round(rho_s, 6)))
    save_file(y, y_base_path+"y_coll_{}".format(round(rho_s, 6)))
    save_file(t, t_base_path+"t_coll_{}".format(round(rho_s, 6)))










