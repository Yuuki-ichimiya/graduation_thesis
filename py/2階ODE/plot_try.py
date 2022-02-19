from import_summary import *
from function import *  # 各設定

x_list = np.array([h * i for i in range(n)])
y_runge4 = np.zeros(n)

if number == 6:
    h_runge = 0.001  # 刻み幅
    num = int(h/h_runge)
    print(num)
    with open('触っちゃいけない/runge4_vanderpol_mu=' + str(mu) + '.csv', newline='') as f:
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
print(y_exact.shape, y_runge4.shape)
if number == 6:
    plt.plot(x_list, y_exact, label="Initial value")  # 厳密解をプロット
    plt.xlabel("t")
    plt.ylabel("x")
else:
    plt.plot(x_list, y_exact, label="The exact solution")  # 厳密解をプロット
    plt.xlabel("x")
    plt.ylabel("y")

plt.plot(x_list, y_runge4, label="Runge-Kutta method(fourth-order)")  # 4次のルンゲクッタ法(厳密)で求めた近似解をプロット

plt.title("Solution Behavior in each solution method")
plt.grid(color="black", linestyle='--', linewidth=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()

##############################################################
fig = plt.figure(figsize=(10, 6))
if number == 6:
    plt.xlabel("t")
    plt.ylabel("x")

else:
    plt.xlabel("x")
    plt.ylabel("y")

plt.plot(x_list, abs(y_runge4 - y_exact), label="Runge-Kutta method(fourth-order)")  # 4次のルンゲクッタ法(厳密)で求めた近似解をプロット
plt.title("Relative error in each solution method")
plt.grid(color="black", linestyle='--', linewidth=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
fig.savefig(
    "触っちゃいけない/mu=" + str(mu) + ".png", bbox_inches="tight")
plt.show()
