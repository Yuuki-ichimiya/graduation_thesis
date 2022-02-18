import numpy as np

from import_summary import *
from utility import *  # 計算基底に関する確率分布を表示
from HHL_function import *  # HHL algorithmに必要な関数
from copy import copy
import csv
import tqdm


# 設定
np.set_printoptions(linewidth=200)
number = 6
if number == 6:
    # d^2y/dt^2=-μ(y^2-1)(dy/dt)-y,y(0)=0, y'(0)=0
    mu = 0.7

    h = 0.0001  # 刻み幅(h=0.05,n=400,nh=20)
    n = 200000  # 考える区間[x_first, x_first + h*(n-1)]
    e = 5
    y_order = 2
    z_order = 1
    x_list = np.array([h * i for i in range(n)])
    y_0 = 0.5  # y_0=a_0
    temp_func = (4 / y_0 ** 2 - 1) * np.exp(-mu * x_list)  # (4/a^2_0-1)exp(-μt)*cos(t)
    y_exact = 2 / np.sqrt(1 + temp_func) * np.cos(x_list)  # y(0)=0.5,y'(0)=0
    diff_y_exact = -2 / np.sqrt(1 + temp_func) * np.sin(x_list)
    y_first = copy(y_exact)
    diff_y_first = copy(diff_y_exact)


    def function(x, axis):
        # dy/dt = z
        func = np.zeros((1 + y_order, 1 + z_order))
        if axis == "y":
            func[0][1] = 1
            return func

        # dz/dt=-μ(y^2-1)z-y=-y+μz-μy^2z
        elif axis == "z":
            func = np.zeros((1 + y_order, 1 + z_order))
            func[1][0] = -1
            func[0][1] = mu
            func[2][1] = -mu

            return func

        else:
            print("unknown axis")
            sys.exit(1)

else:
    print("Invalid number.")
    sys.exit(1)

###################### Runge-kutta法(4次)専用 ######################
y_runge4 = np.zeros(len(x_list))  # y(x)
z_runge4 = np.zeros(len(x_list))  # z:=dy/dx
y_runge4[0] = y_first[0]
z_runge4[0] = diff_y_first[0]

for i in tqdm.tqdm(range(len(x_list) - 1)):
    k_1, l_1 = 0, 0
    for a in range(1 + y_order):
        for b in range(1 + z_order):
            k_1 += h * function(x_list[i], "y")[a][b] * (y_runge4[i] ** a) * (z_runge4[i] ** b)
            l_1 += h * function(x_list[i], "z")[a][b] * (y_runge4[i] ** a) * (z_runge4[i] ** b)

    k_2, l_2 = 0, 0
    for a in range(1 + y_order):
        for b in range(1 + z_order):
            k_2 += h * function(x_list[i] + h / 2, "y")[a][b] * ((y_runge4[i] + k_1 / 2) ** a) * (
                    (z_runge4[i] + l_1 / 2) ** b)
            l_2 += h * function(x_list[i] + h / 2, "z")[a][b] * ((y_runge4[i] + k_1 / 2) ** a) * (
                    (z_runge4[i] + l_1 / 2) ** b)

    k_3, l_3 = 0, 0
    for a in range(1 + y_order):
        for b in range(1 + z_order):
            k_3 += h * function(x_list[i] + h / 2, "y")[a][b] * ((y_runge4[i] + k_2 / 2) ** a) * (
                    (z_runge4[i] + l_2 / 2) ** b)
            l_3 += h * function(x_list[i] + h / 2, "z")[a][b] * ((y_runge4[i] + k_2 / 2) ** a) * (
                    (z_runge4[i] + l_2 / 2) ** b)

    k_4, l_4 = 0, 0
    for a in range(1 + y_order):
        for b in range(1 + z_order):
            k_4 += h * function(x_list[i] + h, "y")[a][b] * ((y_runge4[i] + k_3) ** a) * ((z_runge4[i] + l_3) ** b)
            l_4 += h * function(x_list[i] + h, "z")[a][b] * ((y_runge4[i] + k_3) ** a) * ((z_runge4[i] + l_3) ** b)

    k = (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
    l = (l_1 + 2 * l_2 + 2 * l_3 + l_4) / 6

    y_runge4[i + 1] = y_runge4[i] + k
    z_runge4[i + 1] = z_runge4[i] + l

###################### Runge-kutta法(4次)専用 ######################

with open('runge4_vanderpol_mu=' + str(mu) + '.csv', 'w') as f:
    writer = csv.writer(f)
    for i in range(n):
        writer.writerow([x_list[i], y_runge4[i]])


y_0 = 0.5 # y_0=a_0
temp_func = (4/y_0**2 - 1)*np.exp(-mu *x_list)  # (4/a^2_0-1)exp(-μt)*cos(t)
kinzi = 2/np.sqrt(1+temp_func)*np.cos(x_list)  # y(0)=0.5,y'(0)=0

fig = plt.figure(figsize=(10, 4.8))
plt.plot(x_list, y_runge4, label="Runge-Kutta method")  # Runge-Kutta法で求めた近似解をプロット
plt.plot(x_list, y_exact, label="kinzi")  # Runge-Kutta法で求めた近似解をプロット
plt.title("Solution Behavior in each solution method(mu=" + str(mu) + ")")
plt.grid(color="black", linestyle='--', linewidth=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xlabel("x")
plt.ylabel("y")
fig.savefig("資料/plot_mu=" + str(mu) + ".png")
plt.show()

fig = plt.figure(figsize=(10, 4.8))
plt.plot(x_list, abs(y_exact - y_runge4), label="Runge-Kutta method")  # Runge-Kutta法で求めた近似解をプロット
plt.title("Error(mu=" + str(mu) + ")")
plt.grid(color="black", linestyle='--', linewidth=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xlabel("x")
plt.ylabel("y")
fig.savefig("資料/error_mu=" + str(mu) + ".png")
plt.show()

