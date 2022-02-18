from import_summary import *
from function import *  # 各設定

x_list = np.array([h * i for i in range(n)])
HHL_runge_result = np.zeros((len(reg_nbit_list), n))
y_exact_newton = np.zeros(n)
y_runge4 = np.zeros(n)

for index, reg_nbit in enumerate(reg_nbit_list):
    with open("2階ODE数値結果/HHL(number=" + str(number) + ",reg_qubits=" + str(reg_nbit) + ",h=" + str(
            h) + ",n=" + str(n) + ").csv", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            HHL_runge_result[index][i] = row[1]

with open("2階ODE数値結果/NEWTON_EXACT(number=" + str(number) + ",h=" + str(
        h) + ",n=" + str(n) + ").csv", newline="") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        y_exact_newton[i] = row[1]

if number == 6:
    h_runge = 0.0001  # 刻み幅
    num = int(h / h_runge)
    print(num)
    with open('2階ODE数値結果/runge4_vanderpol_mu=' + str(mu) + '.csv', newline='') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i // num == n:
                break
            elif i % num == 0:
                y_runge4[i // num] = row[1]
            else:
                continue
else:
    with open("2階ODE数値結果/RUNGE_EXACT(number=" + str(number) + ",h=" + str(
            h) + ",n=" + str(n) + ").csv", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            y_runge4[i] = row[1]

fig = plt.figure(figsize=(10, 4.8))
plt.plot(x_list, HHL_runge_result[0],
         label="Newton-Raphson method(HHL,reg=" + str(reg_nbit_list[0]) + ")", color="red")  # オイラー法(厳密)で求めた近似解をプロット
plt.plot(x_list, HHL_runge_result[1],
         label="Newton-Raphson method(HHL,reg=" + str(reg_nbit_list[1]) + ")", color="orange")  # オイラー法(厳密)で求めた近似解をプロット
plt.plot(x_list, HHL_runge_result[2],
         label="Newton-Raphson method(HHL,reg=" + str(reg_nbit_list[2]) + ")", color="blue")  # オイラー法(厳密)で求めた近似解をプロット
plt.plot(x_list, y_exact_newton, label="Newton-Raphson(exact)", color="green")  # オイラー法(HHL)で求めた近似解をプロット

if number == 6:
    plt.plot(x_list, y_exact, label="Initial value", linestyle="dashed", color="black")  # 厳密解をプロット
    plt.xlabel("t")
    plt.ylabel("x")
else:
    plt.plot(x_list, y_runge4, label="Runge-Kutta method(fourth-order)")  # 4次のルンゲクッタ法(厳密)で求めた近似解をプロット
    plt.plot(x_list, y_exact, label="The exact solution")  # 厳密解をプロット
    plt.xlabel("x")
    plt.ylabel("y")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='large')
plt.title("Solution Behavior in each solution method")
plt.grid(color="black", linestyle='--', linewidth=0.3)
fig.savefig("2階ODE数値結果/plot(number=" + str(number) + ",reg_qubits=" + str(reg_nbit_list) + ",h=" + str(
    h) + ",n=" + str(n) + ").png", bbox_inches="tight")
plt.show()

##############################################################
fig = plt.figure(figsize=(10, 6))
plt.plot(x_list, abs((HHL_runge_result[0] - y_runge4) / y_runge4),
         label="Newton-Raphson method(HHL,reg=" + str(reg_nbit_list[0]) + ")", color="red")  # オイラー法(厳密)で求めた近似解をプロット
plt.plot(x_list, abs((HHL_runge_result[1] - y_runge4) / y_runge4),
         label="Newton-Raphson method(HHL,reg=" + str(reg_nbit_list[1]) + ")", color="orange")  # オイラー法(厳密)で求めた近似解をプロット
plt.plot(x_list, abs((HHL_runge_result[2] - y_runge4) / y_runge4),
         label="Newton-Raphson method(HHL,reg=" + str(reg_nbit_list[2]) + ")", color="blue")  # オイラー法(厳密)で求めた近似解をプロット
plt.plot(x_list, abs((y_exact_newton - y_runge4) / y_runge4),
         label="Newton-Raphson method(exact)", color="green")  # オイラー法(HHL)で求めた近似解をプロット

if number == 6:
    plt.xlabel("t")
    plt.ylabel("x")

else:
    plt.plot(x_list, abs((y_runge4 - y_exact) / y_exact),
             label="Runge-Kutta method(fourth-order)", color="brown")  # 4次のルンゲクッタ法(厳密)で求めた近似解をプロット
    plt.xlabel("x")
    plt.ylabel("y")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='large')
plt.title("Relative error in each solution method")
plt.grid(color="black", linestyle='--', linewidth=0.3)
plt.yscale("log")
fig.savefig(
    "2階ODE数値結果/error(number=" + str(number) + ",reg_qubits=" + str(reg_nbit_list) + ",h=" + str(
        h) + ",n=" + str(n) + ").png", bbox_inches="tight")
plt.show()
