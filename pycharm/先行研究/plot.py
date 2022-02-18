from import_summary import *
from function import *  # 各設定

x_list = np.array([h * i for i in range(n)])
exact_euler_result = np.zeros(n)
HHL_euler_result = np.zeros(n)
exact_runge_result = np.zeros(n)
HHL_runge_result = np.zeros(n)

with open("先行研究数値結果/Euler_EXACT(number=" + str(number) + ",reg_qubits=" + str(reg_nbit) + ",h=" + str(
        h) + ",n=" + str(n) + ").csv", newline="") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        exact_euler_result[i] = row[1]

with open("先行研究数値結果/Euler_HHL(number=" + str(number) + ",reg_qubits=" + str(reg_nbit) + ",h=" + str(
        h) + ",n=" + str(n) + ").csv", newline="") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        HHL_euler_result[i] = row[1]

with open("先行研究数値結果/Runge_EXACT(number=" + str(number) + ",reg_qubits=" + str(reg_nbit) + ",h=" + str(
        h) + ",n=" + str(n) + ").csv", newline="") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        exact_runge_result[i] = row[1]

with open("先行研究数値結果/Runge_HHL(number=" + str(number) + ",reg_qubits=" + str(reg_nbit) + ",h=" + str(
        h) + ",n=" + str(n) + ").csv", newline="") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        HHL_runge_result[i] = row[1]


fig = plt.figure(figsize=(10, 4.8))
plt.plot(x_list, y_exact, label="The exact solution", color="black")  # 厳密解をプロット
plt.plot(x_list, exact_euler_result, label="Euler method(exact)", color="red")  # オイラー法(厳密)で求めた近似解をプロット
plt.plot(x_list, HHL_euler_result, label="Euler method(HHL)", color="orange")  # オイラー法(HHL)で求めた近似解をプロット
plt.plot(x_list, exact_runge_result, label="Runge-Kutta method(exact,fourth-order)",
         color="blue")  # 4次のルンゲクッタ法(厳密)で求めた近似解をプロット
plt.plot(x_list, HHL_runge_result, label="Runge-Kutta method(HHL,fourth-order)",
         color="brown")  # 4次のルンゲクッタ法(HHL)で求めた近似解をプロット

plt.title("Solution Behavior in each solution method")
plt.grid(color="black", linestyle='--', linewidth=0.3)
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', fontsize='large')
plt.xlabel("x")
plt.ylabel("y")
fig.savefig("先行研究数値結果/plot(number=" + str(number) + ",reg_qubits=" + str(reg_nbit) + ",h=" + str(
    h) + ",n=" + str(n) + ").png", bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(10, 6))
plt.plot(x_list, abs((exact_euler_result - y_exact) / y_exact), label="Euler method(exact)",
         color="red")  # オイラー法(厳密)で求めた近似解をプロット
plt.plot(x_list, abs((HHL_euler_result - y_exact) / y_exact), label="Euler method(HHL)",
         color="orange")  # オイラー法(HHL)で求めた近似解をプロット
plt.plot(x_list, abs((exact_runge_result - y_exact) / y_exact),
         label="Runge-Kutta method(exact,four-order)", color="blue")  # 2次のルンゲクッタ法(厳密)で求めた近似解をプロット
plt.plot(x_list, abs((HHL_runge_result - y_exact) / y_exact),
         label="Runge-Kutta method(HHL,four-order)", color="brown")  # 2次のルンゲクッタ法(HHL)で求めた近似解をプロット

plt.title("Relative error in each solution method")
plt.grid(color="black", linestyle='--', linewidth=0.3)
plt.legend(bbox_to_anchor=(0.1, 0), loc='lower left', fontsize='large')
plt.xlabel("x")
plt.ylabel("y")
plt.yscale("log")
fig.savefig("先行研究数値結果/error(number=" + str(number) + ",reg_qubits=" + str(reg_nbit) + ",h=" + str(
    h) + ",n=" + str(n) + ").png", bbox_inches="tight")
plt.show()
