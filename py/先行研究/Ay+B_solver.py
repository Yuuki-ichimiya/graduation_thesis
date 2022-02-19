from import_summary import *
from utility import *  # 計算基底に関する確率分布を表示
from HHL_function import *  # 計算基底に関する確率分布を表示
from function import *  # 計算基底に関する確率分布を表示

###################### euler法専用 ######################
# Wx=cに直す
W_euler = np.zeros((n, n))
b_vec_euler = np.zeros(n)
b_vec_euler[0] = y_first

for i in range(n):
    W_euler[i][i] = 1

for i in range(n - 1):
    f_1 = f_function(x_list[i])
    g_1 = g_function(x_list[i])
    W_euler[i + 1][i] = -h * f_1 - 1
    b_vec_euler[i + 1] = h * g_1

# Wは必ずHermiteではないので[[0,W],[W^T,0]]、[b,0]にする
dim = 2 * n
W_euler = np.block([[np.zeros((n, n)), W_euler], [W_euler.T, np.zeros((n, n))]])  # 2n*2n次元の行列
b_vec_euler = np.block([b_vec_euler, np.zeros(n)])  # 2n次元のベクトル

## Wの固有値を確認 -> [-pi, pi] に収まっている
eigenvalue_euler = np.linalg.eigh(W_euler)[0]
for value in eigenvalue_euler:
    if value < -np.pi or np.pi < value:
        print("caution:eigenvalues exist that are not included in [-π,π].")
        exit(1)

start = time.time()
y_HHL_euler = HHL_algorithm(W_euler, b_vec_euler, dim, reg_nbit, scale_fac)
t = time.time() - start

# 厳密解
y_exact_euler = np.linalg.lstsq(W_euler, b_vec_euler, rcond=0)[0]

# 切り抜き
y_HHL_euler = y_HHL_euler[n:2 * n]
y_exact_euler = y_exact_euler[n:2 * n]

with open("先行研究数値結果/Euler_HHL(number=" + str(number) + ",reg_qubits=" + str(reg_nbit) + ",h=" + str(h) + ",n=" + str(
        n) + ").csv", 'w', newline='') as f:
    writer = csv.writer(f)
    for i in range(n):
        writer.writerow([x_list[i], y_HHL_euler[i]])

with open("先行研究数値結果/Euler_EXACT(number=" + str(number) + ",reg_qubits=" + str(reg_nbit) + ",h=" + str(h) + ",n=" + str(
        n) + ").csv", 'w', newline='') as f:
    writer = csv.writer(f)
    for i in range(n):
        writer.writerow([x_list[i], y_exact_euler[i]])

print("Elapsed time[s]:", t)
print("method:euler method")
print("register qubits:", reg_nbit)
print("h:", h, "n:", n)
###################### Euler法実行 ######################

###################### Runge-kutta法(4次)専用 ######################
# Wx=cに直す
W_runge4 = np.zeros((n, n))
b_vec_runge4 = np.zeros(n)
b_vec_runge4[0] = y_first

for i in range(n):
    W_runge4[i][i] = 1

for i in range(n - 1):
    f_1 = f_function(x_list[i])
    f_2 = f_function(x_list[i] + h / 2)
    f_3 = f_function(x_list[i] + h)
    g_1 = g_function(x_list[i])
    g_2 = g_function(x_list[i] + h / 2)
    g_3 = g_function(x_list[i] + h)

    k_1 = np.array([h * g_1, h * f_1])  # k_1
    k_2 = h * (f_2 * (np.array([0, 1]) + k_1 / 2) + np.array([g_2, 0]))
    k_3 = h * (f_2 * (np.array([0, 1]) + k_2 / 2) + np.array([g_2, 0]))
    k_4 = h * (f_3 * (np.array([0, 1]) + k_3) + np.array([g_3, 0]))
    k = (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

    W_runge4[i + 1][i] = -(k[1] + 1)
    b_vec_runge4[i + 1] = k[0]

# Wは必ずHermiteではないので[[0,W],[W^T,0]]、[b,0]にする
dim = 2 * n
W_runge4 = np.block([[np.zeros((n, n)), W_runge4], [W_runge4.T, np.zeros((n, n))]])  # 2n*2n次元の行列
b_vec_runge4 = np.block([b_vec_runge4, np.zeros(n)])  # 2n次元のベクトル

## Wの固有値を確認 -> [-pi, pi] に収まっている
eigenvalue_runge4 = np.linalg.eigh(W_runge4)[0]
for value in eigenvalue_runge4:
    if value < -np.pi or np.pi < value:
        print("caution:eigenvalues exist that are not included in [-π,π].")
        exit(1)

start = time.time()
y_HHL_runge4 = HHL_algorithm(W_runge4, b_vec_runge4, dim, reg_nbit, scale_fac)
t = time.time() - start

# 厳密解
y_exact_runge4 = np.linalg.lstsq(W_runge4, b_vec_runge4, rcond=0)[0]

# 切り抜き
y_HHL_runge4 = y_HHL_runge4[n:2 * n]
y_exact_runge4 = y_exact_runge4[n:2 * n]

with open("先行研究数値結果/Runge_HHL(number=" + str(number) + ",reg_qubits=" + str(reg_nbit) + ",h=" + str(h) + ",n=" + str(
        n) + ").csv", 'w', newline='') as f:
    writer = csv.writer(f)
    for i in range(n):
        writer.writerow([x_list[i], y_HHL_runge4[i]])

with open("先行研究数値結果/Runge_EXACT(number=" + str(number) + ",reg_qubits=" + str(reg_nbit) + ",h=" + str(h) + ",n=" + str(
        n) + ").csv", 'w', newline='') as f:
    writer = csv.writer(f)
    for i in range(n):
        writer.writerow([x_list[i], y_exact_runge4[i]])

print("Elapsed time[s]:", t)
print("method:Runge-Kutta method(fourth order)")
print("register qubits:", reg_nbit)
print("h:", h, "n:", n)
###################### Runge-kutta法(4次)実行 ######################