from import_summary import *
from utility import *  # 計算基底に関する確率分布を表示
from HHL_function import *  # HHL algorithmに必要な関数
from function import *  # 各設定
import tqdm


# 多項式(の係数)list1と多項式(の係数)list2の掛け算、十分要素数が確保されているならenough=1
def times_poly(poly1, poly2, enough=1):
    result = np.zeros((poly1.shape[0] + poly2.shape[0], poly1.shape[1] + poly2.shape[1]))
    for i in range(poly1.shape[0]):
        for j in range(poly1.shape[1]):
            for m in range(poly2.shape[0]):
                for n in range(poly2.shape[1]):
                    result[i + m][j + n] += poly1[i][j] * poly2[m][n]

    # list_1とlist_2の要素数は同じで、かつ配列の長さが十分のとき
    if enough == 1:
        result = np.array([i[:poly1.shape[1]] for i in result[:poly1.shape[0]]])

    return result


# 多項式(の係数)listの微分
def differential(poly, axis):
    result = np.zeros((poly.shape[0], poly.shape[1]))
    if axis == "y":
        for i in range(poly.shape[0] - 1):
            for j in range(poly.shape[1]):
                result[i][j] = (i + 1) * poly[i + 1][j]

        return result

    elif axis == "z":
        for i in range(poly.shape[0]):
            for j in range(poly.shape[1] - 1):
                result[i][j] = (j + 1) * poly[i][j + 1]

        return result

    else:
        print("axis error")
        sys.exit(1)


###################### Runge-kutta法(4次)専用 ######################
y_runge4 = np.zeros(len(x_list))  # y(x)
z_runge4 = np.zeros(len(x_list))  # z:=dy/dx
y_runge4[0] = y_first[0]
z_runge4[0] = diff_y_first[0]

start = time.time()
for i in range(len(x_list) - 1):
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

t = time.time() - start

with open("2階ODE数値結果/RUNGE_EXACT(number=" + str(number) + ",h=" + str(h) + ",n=" + str(n) + ").csv", 'w',
          newline='') as f:
    writer = csv.writer(f)
    for i in range(n):
        writer.writerow([x_list[i], y_runge4[i]])

print("Runge-Kutta method calculation completed.")
print("Elapsed time[s]:", t)
print("number:", number, "h:", h, " n:", n, "\n")
###################### Runge-kutta法(4次)専用 ######################
del y_runge4
del z_runge4


###################### Runge-kutta法(4次・Newton)専用 ######################

def jacobi_newton(x, diff_y_k_coef_hybr, diff_z_k_coef_hybr, diff_y_l_coef_hybr, diff_z_l_coef_hybr):
    jacobi = np.eye(2 * n)
    for i in range(n - 1):
        jacobi[i + 1][i] = -1
        for y in range(1 + max_ydim):
            for z in range(1 + max_zdim):
                jacobi[i + 1][i] += -diff_y_k_coef_hybr[i][y][z] * (x[i] ** y) * (x[n + i] ** z)
                jacobi[i + 1][n + i] += -diff_z_k_coef_hybr[i][y][z] * (x[i] ** y) * (x[n + i] ** z)

    for i in range(n - 1):
        jacobi[n + i + 1][n + i] = -1
        for y in range(1 + max_ydim):
            for z in range(1 + max_zdim):
                jacobi[n + i + 1][n + i] += -diff_z_l_coef_hybr[i][y][z] * (x[i] ** y) * (x[n + i] ** z)
                jacobi[n + i + 1][i] += -diff_y_l_coef_hybr[i][y][z] * (x[i] ** y) * (x[n + i] ** z)

    return jacobi


def right_side_newton(x, k_coef_newton, l_coef_newton):
    formula = np.array([0])
    for i in range(n - 1):
        k_poly = 0
        for y in range(1 + max_ydim):
            for z in range(1 + max_zdim):
                k_poly += k_coef_newton[i][y][z] * (x[i] ** y) * (x[n + i] ** z)

        formula = np.append(formula, -x[i + 1] + x[i] + k_poly)  # (Jacobi行列)*解 = -(y_(n+1)-y_(n)+k)

    formula = np.append(formula, 0)
    for i in range(n - 1):
        l_poly = 0
        for y in range(1 + max_ydim):
            for z in range(1 + max_zdim):
                l_poly += l_coef_newton[i][y][z] * (x[i] ** y) * (x[n + i] ** z)

        formula = np.append(formula, -x[n + i + 1] + x[n + i] + l_poly)  # (Jacobi行列)*解 = -(z_(n+1)-z_(n)+l)

    return formula


# newton_first = copy(y_first) # 初期条件
# diff_newton_first = copy(diff_y_exact)
# for i in range(1, len(newton_first)):
#     newton_first[i] += random.uniform(e, -e)

newton_first = np.array([y_first[0] for i in range(len(x_list))])  # 初期条件
diff_newton_first = np.array([diff_y_first[0] for i in range(len(x_list))])  # 初期条件

# y_iの最大次数は1 + max_ydim
# z_iの最大次数は1 + max_zdim
k_coef_newton = np.zeros((n, 1 + max_ydim, 1 + max_zdim))
l_coef_newton = np.zeros((n, 1 + max_ydim, 1 + max_zdim))
diff_y_k_coef_newton = np.zeros((n, 1 + max_ydim, 1 + max_zdim))
diff_z_k_coef_newton = np.zeros((n, 1 + max_ydim, 1 + max_zdim))
diff_y_l_coef_newton = np.zeros((n, 1 + max_ydim, 1 + max_zdim))
diff_z_l_coef_newton = np.zeros((n, 1 + max_ydim, 1 + max_zdim))
f_value = np.zeros((1 + y_order, 1 + z_order, 3))
g_value = np.zeros((1 + y_order, 1 + z_order, 3))

start = time.time()
print("k,lの準備")
for i in range(n - 1):
    for y in range(1 + y_order):
        for z in range(1 + z_order):
            for m in range(3):
                f_value[y][z][m] = function(x_list[i] + m * h / 2, "y")[y][z]  # f(x+m*h/2,y,z)のy^(.)z^(.)の係数
                g_value[y][z][m] = function(x_list[i] + m * h / 2, "z")[y][z]  # g(x+m*h/2,y,z)のy^(.)z^(.)の係数

    k_1 = np.zeros((1 + max_ydim, 1 + max_zdim))
    k_2 = np.zeros((1 + max_ydim, 1 + max_zdim))
    k_3 = np.zeros((1 + max_ydim, 1 + max_zdim))
    k_4 = np.zeros((1 + max_ydim, 1 + max_zdim))
    k = np.zeros((1 + max_ydim, 1 + max_zdim))

    l_1 = np.zeros((1 + max_ydim, 1 + max_zdim))
    l_2 = np.zeros((1 + max_ydim, 1 + max_zdim))
    l_3 = np.zeros((1 + max_ydim, 1 + max_zdim))
    l_4 = np.zeros((1 + max_ydim, 1 + max_zdim))
    l = np.zeros((1 + max_ydim, 1 + max_zdim))
    ################## k_1,l_1はここから ##################
    for y in range(1 + y_order):
        for z in range(1 + z_order):
            k_1[y][z] = h * f_value[y][z][0]

    for y in range(1 + y_order):
        for z in range(1 + z_order):
            l_1[y][z] = h * g_value[y][z][0]

    k += k_1 / 6.0
    l += l_1 / 6.0

    ################## k_1,l_1はここまで ##################
    ################## k_2,l_2はここから ##################
    k_1 = k_1 / 2
    k_1[1][0] += 1  # y+k_1/2
    l_1 = l_1 / 2
    l_1[0][1] += 1  # z+k_2/2
    k_temp = np.zeros((1 + y_order, 1 + max_ydim, 1 + max_zdim))
    l_temp = np.zeros((1 + z_order, 1 + max_ydim, 1 + max_zdim))
    k_temp[0][0][0] = 1  # (y+k_1/2)^0
    l_temp[0][0][0] = 1  # (z+k_1/2)^0
    for a in range(y_order):
        k_temp[a + 1] = times_poly(k_temp[a], k_1)  # k_temp[i]=(y+k_1/2)^i

    for b in range(z_order):
        l_temp[b + 1] = times_poly(l_temp[b], l_1)  # l_temp[i]=(z+l_1/2)^i

    for a in range(1 + y_order):
        for b in range(1 + z_order):
            k_2 += h * f_value[a][b][1] * times_poly(k_temp[a], l_temp[b])
            l_2 += h * g_value[a][b][1] * times_poly(k_temp[a], l_temp[b])

    k += k_2 / 3.0
    l += l_2 / 3.0

    ################## k_2,l_2はここまで ##################
    ################## k_3,l_3はここから ##################
    k_2 = k_2 / 2
    k_2[1][0] += 1  # y+k_2/2
    l_2 = l_2 / 2
    l_2[0][1] += 1  # z+k_2/2
    k_temp = np.zeros((1 + y_order, 1 + max_ydim, 1 + max_zdim))
    l_temp = np.zeros((1 + z_order, 1 + max_ydim, 1 + max_zdim))
    k_temp[0][0][0] = 1  # (y+k_2/2)^0
    l_temp[0][0][0] = 1  # (z+k_2/2)^0
    for a in range(y_order):
        k_temp[a + 1] = times_poly(k_temp[a], k_2)  # k_temp[i]=(y+k_2/2)^i

    for b in range(z_order):
        l_temp[b + 1] = times_poly(l_temp[b], l_2)  # l_temp[i]=(z+l_2/2)^i

    for a in range(1 + y_order):
        for b in range(1 + z_order):
            k_3 += h * f_value[a][b][1] * times_poly(k_temp[a], l_temp[b])
            l_3 += h * g_value[a][b][1] * times_poly(k_temp[a], l_temp[b])

    k += k_3 / 3.0
    l += l_3 / 3.0

    ################## k_3,l_3はここまで ##################
    ################## k_4,l_4はここから ##################
    k_3[1][0] += 1  # y+k_3
    l_3[0][1] += 1  # z+k_3
    k_temp = np.zeros((1 + y_order, 1 + max_ydim, 1 + max_zdim))
    l_temp = np.zeros((1 + z_order, 1 + max_ydim, 1 + max_zdim))
    k_temp[0][0][0] = 1  # (y+k_2/2)^0
    l_temp[0][0][0] = 1  # (z+k_2/2)^0
    for a in range(y_order):
        k_temp[a + 1] = times_poly(k_temp[a], k_3)  # k_temp[i]=(y+k_3)^i

    for b in range(z_order):
        l_temp[b + 1] = times_poly(l_temp[b], l_3)  # l_temp[i]=(z+l_3)^i

    for a in range(1 + y_order):
        for b in range(1 + z_order):
            k_4 += h * f_value[a][b][2] * times_poly(k_temp[a], l_temp[b])  # h*f_{ij}*(y+k_3)^i(z+l_3)^j
            l_4 += h * g_value[a][b][2] * times_poly(k_temp[a], l_temp[b])  # h*g_{ij}*(y+k_3)^i(z+l_3)^j

    k += k_4 / 6.0
    l += l_4 / 6.0
    ################## k_4,l_4はここまで ##################

    k_coef_newton[i] = k  # kを多項式で表示したときの係数
    l_coef_newton[i] = l  # lを多項式で表示したときの係数
    diff_y_k_coef_newton[i] = differential(k, "y")  # dk/dyを多項式で表示したときの係数
    diff_z_k_coef_newton[i] = differential(k, "z")  # dk/dyを多項式で表示したときの係数
    diff_y_l_coef_newton[i] = differential(l, "y")  # dl/dyを多項式で表示したときの係数
    diff_z_l_coef_newton[i] = differential(l, "z")  # dl/dyを多項式で表示したときの係数

print("k,lの作成終了")
for _ in tqdm.tqdm(range(depth)):
    # 係数行列の生成(Jacobi行列)
    jacobi_matrix_newton = jacobi_newton(np.concatenate([newton_first, diff_newton_first]), diff_y_k_coef_newton,
                                         diff_z_k_coef_newton, diff_y_l_coef_newton, diff_z_l_coef_newton)
    # 右辺の生成
    right_newton = right_side_newton(np.concatenate([newton_first, diff_newton_first]), k_coef_newton, l_coef_newton)
    Delta = np.linalg.lstsq(jacobi_matrix_newton, right_newton, rcond=0)[0]

    newton_first += Delta[:n]
    diff_newton_first += Delta[n:2 * n]
    print(np.average(np.abs(Delta[:n])))

t = time.time() - start
y_exact_newton = newton_first

with open("2階ODE数値結果/NEWTON_EXACT(number=" + str(number) + ",h=" + str(h) + ",n=" + str(n) + ").csv", 'w',
          newline='') as f:
    writer = csv.writer(f)
    for i in range(n):
        writer.writerow([x_list[i], y_exact_newton[i]])

print("Newton calculation completed.")
print("Elapsed time[s]:", t)
print("number:", number, "h:", h, " n:", n, "\n")

###################### Runge-kutta法(4次・Newton)専用 ######################

del k_coef_newton
del l_coef_newton
del diff_y_k_coef_newton
del diff_z_k_coef_newton
del diff_y_l_coef_newton
del diff_z_l_coef_newton


def jacobi_HHL(x, diff_y_k_coef_HHL, diff_z_k_coef_HHL, diff_y_l_coef_HHL, diff_z_l_coef_HHL):
    jacobi = np.eye(2 * n)
    for i in range(n - 1):
        jacobi[i + 1][i] = -1
        for y in range(1 + max_ydim):
            for z in range(1 + max_zdim):
                jacobi[i + 1][i] += -diff_y_k_coef_HHL[i][y][z] * (x[i] ** y) * (x[n + i] ** z)
                jacobi[i + 1][n + i] += -diff_z_k_coef_HHL[i][y][z] * (x[i] ** y) * (x[n + i] ** z)

    for i in range(n - 1):
        jacobi[n + i + 1][n + i] = -1
        for y in range(1 + max_ydim):
            for z in range(1 + max_zdim):
                jacobi[n + i + 1][n + i] += -diff_z_l_coef_HHL[i][y][z] * (x[i] ** y) * (x[n + i] ** z)
                jacobi[n + i + 1][i] += -diff_y_l_coef_HHL[i][y][z] * (x[i] ** y) * (x[n + i] ** z)

    return jacobi


def right_side_HHL(x, k_coef_HHL, l_coef_HHL):
    formula = np.array([0])
    for i in range(n - 1):
        k_poly = 0
        for y in range(1 + max_ydim):
            for z in range(1 + max_zdim):
                k_poly += k_coef_HHL[i][y][z] * (x[i] ** y) * (x[n + i] ** z)

        formula = np.append(formula, -x[i + 1] + x[i] + k_poly)  # (Jacobi行列)*解 = -(y_(n+1)-y_(n)+k)

    formula = np.append(formula, 0)
    for i in range(n - 1):
        l_poly = 0
        for y in range(1 + max_ydim):
            for z in range(1 + max_zdim):
                l_poly += l_coef_HHL[i][y][z] * (x[i] ** y) * (x[n + i] ** z)

        formula = np.append(formula, -x[n + i + 1] + x[n + i] + l_poly)  # (Jacobi行列)*解 = -(z_(n+1)-z_(n)+l)

    return formula


###################### Runge-kutta法(4次・HHL)専用 ######################
# HHL_first = copy(y_first)  # 初期条件
# diff_HHL_first = copy(diff_y_exact)  # 初期条件
# for i in range(1, len(HHL_first)):
#     HHL_first[i] += random.uniform(e, -e)

HHL_runge_result = np.zeros((len(reg_nbit_list), n))
k_coef_HHL = np.zeros((n, 1 + max_ydim, 1 + max_zdim))
l_coef_HHL = np.zeros((n, 1 + max_ydim, 1 + max_zdim))
diff_y_k_coef_HHL = np.zeros((n, 1 + max_ydim, 1 + max_zdim))
diff_z_k_coef_HHL = np.zeros((n, 1 + max_ydim, 1 + max_zdim))
diff_y_l_coef_HHL = np.zeros((n, 1 + max_ydim, 1 + max_zdim))
diff_z_l_coef_HHL = np.zeros((n, 1 + max_ydim, 1 + max_zdim))
f_value = np.zeros((1 + y_order, 1 + z_order, 3))
g_value = np.zeros((1 + y_order, 1 + z_order, 3))

print("k,lの作成準備")
for i in range(n - 1):
    for y in range(1 + y_order):
        for z in range(1 + z_order):
            for m in range(3):
                f_value[y][z][m] = function(x_list[i] + m * h / 2, "y")[y][z]  # f(x+m*h/2,y,z)のy^(.)z^(.)の係数
                g_value[y][z][m] = function(x_list[i] + m * h / 2, "z")[y][z]  # g(x+m*h/2,y,z)のy^(.)z^(.)の係数

    k_1 = np.zeros((1 + max_ydim, 1 + max_zdim))
    k_2 = np.zeros((1 + max_ydim, 1 + max_zdim))
    k_3 = np.zeros((1 + max_ydim, 1 + max_zdim))
    k_4 = np.zeros((1 + max_ydim, 1 + max_zdim))
    k = np.zeros((1 + max_ydim, 1 + max_zdim))

    l_1 = np.zeros((1 + max_ydim, 1 + max_zdim))
    l_2 = np.zeros((1 + max_ydim, 1 + max_zdim))
    l_3 = np.zeros((1 + max_ydim, 1 + max_zdim))
    l_4 = np.zeros((1 + max_ydim, 1 + max_zdim))
    l = np.zeros((1 + max_ydim, 1 + max_zdim))
    ################## k_1,l_1はここから ##################
    for y in range(1 + y_order):
        for z in range(1 + z_order):
            k_1[y][z] = h * f_value[y][z][0]

    for y in range(1 + y_order):
        for z in range(1 + z_order):
            l_1[y][z] = h * g_value[y][z][0]

    k += k_1 / 6.0
    l += l_1 / 6.0

    ################## k_1,l_1はここまで ##################
    ################## k_2,l_2はここから ##################
    k_1 = k_1 / 2
    k_1[1][0] += 1  # y+k_1/2
    l_1 = l_1 / 2
    l_1[0][1] += 1  # z+k_2/2
    k_temp = np.zeros(
        (1 + y_order, 1 + max_ydim,
         1 + max_zdim))
    l_temp = np.zeros(
        (1 + z_order, 1 + max_ydim,
         1 + max_zdim))
    k_temp[0][0][0] = 1  # (y+k_1/2)^0
    l_temp[0][0][0] = 1  # (z+k_1/2)^0
    for a in range(y_order):
        k_temp[a + 1] = times_poly(k_temp[a], k_1)  # k_temp[i]=(y+k_1/2)^i

    for b in range(z_order):
        l_temp[b + 1] = times_poly(l_temp[b], l_1)  # l_temp[i]=(z+l_1/2)^i

    for a in range(1 + y_order):
        for b in range(1 + z_order):
            k_2 += h * f_value[a][b][1] * times_poly(k_temp[a], l_temp[b])
            l_2 += h * g_value[a][b][1] * times_poly(k_temp[a], l_temp[b])

    k += k_2 / 3.0
    l += l_2 / 3.0

    ################## k_2,l_2はここまで ##################
    ################## k_3,l_3はここから ##################
    k_2 = k_2 / 2
    k_2[1][0] += 1  # y+k_2/2
    l_2 = l_2 / 2
    l_2[0][1] += 1  # z+k_2/2
    k_temp = np.zeros((1 + y_order, 1 + max_ydim, 1 + max_zdim))
    l_temp = np.zeros((1 + z_order, 1 + max_ydim, 1 + max_zdim))
    k_temp[0][0][0] = 1  # (y+k_2/2)^0
    l_temp[0][0][0] = 1  # (z+k_2/2)^0
    for a in range(y_order):
        k_temp[a + 1] = times_poly(k_temp[a], k_2)  # k_temp[i]=(y+k_2/2)^i

    for b in range(z_order):
        l_temp[b + 1] = times_poly(l_temp[b], l_2)  # l_temp[i]=(z+l_2/2)^i

    for a in range(1 + y_order):
        for b in range(1 + z_order):
            k_3 += h * f_value[a][b][1] * times_poly(k_temp[a], l_temp[b])
            l_3 += h * g_value[a][b][1] * times_poly(k_temp[a], l_temp[b])

    k += k_3 / 3.0
    l += l_3 / 3.0

    ################## k_3,l_3はここまで ##################
    ################## k_4,l_4はここから ##################
    k_3[1][0] += 1  # y+k_3
    l_3[0][1] += 1  # z+k_3
    k_temp = np.zeros((1 + y_order, 1 + max_ydim, 1 + max_zdim))
    l_temp = np.zeros((1 + z_order, 1 + max_ydim, 1 + max_zdim))
    k_temp[0][0][0] = 1  # (y+k_2/2)^0
    l_temp[0][0][0] = 1  # (z+k_2/2)^0
    for a in range(y_order):
        k_temp[a + 1] = times_poly(k_temp[a], k_3)  # k_temp[i]=(y+k_3)^i

    for b in range(z_order):
        l_temp[b + 1] = times_poly(l_temp[b], l_3)  # l_temp[i]=(z+l_3)^i

    for a in range(1 + y_order):
        for b in range(1 + z_order):
            k_4 += h * f_value[a][b][2] * times_poly(k_temp[a], l_temp[b])  # h*f_{ij}*(y+k_3)^i(z+l_3)^j
            l_4 += h * g_value[a][b][2] * times_poly(k_temp[a], l_temp[b])  # h*g_{ij}*(y+k_3)^i(z+l_3)^j

    k += k_4 / 6.0
    l += l_4 / 6.0
    ################## k_4,l_4はここまで ##################

    k_coef_HHL[i] = k  # kを多項式で表示したときの係数
    l_coef_HHL[i] = l  # lを多項式で表示したときの係数
    diff_y_k_coef_HHL[i] = differential(k, "y")  # dk/dyを多項式で表示したときの係数
    diff_z_k_coef_HHL[i] = differential(k, "z")  # dk/dyを多項式で表示したときの係数
    diff_y_l_coef_HHL[i] = differential(l, "y")  # dl/dyを多項式で表示したときの係数
    diff_z_l_coef_HHL[i] = differential(l, "z")  # dl/dyを多項式で表示したときの係数

print("k,lの作成終了")

for index, reg_nbit in enumerate(reg_nbit_list):
    HHL_first = np.array([y_first[0] for i in range(len(x_list))])  # 初期条件
    diff_HHL_first = np.array([diff_y_first[0] for i in range(len(x_list))])  # 初期条件

    # Wx=cに直す
    start = time.time()
    for _ in tqdm.tqdm(range(depth)):

        J_HHL = jacobi_HHL(np.concatenate([HHL_first, diff_HHL_first]), diff_y_k_coef_HHL, diff_z_k_coef_HHL,
                           diff_y_l_coef_HHL, diff_z_l_coef_HHL)  # Jacobi行列を計算
        b_HHL = right_side_HHL(np.concatenate([HHL_first, diff_HHL_first]), k_coef_HHL, l_coef_HHL)  # 連立方程式の右辺

        # Wは必ずHermiteではないので[[0,W],[W^T,0]]、[b,0]にする
        dim = 2 * (2 * n)  # もともと2n次元でHermiteにすると4n次元になる
        J_HHL = np.block([[np.zeros((2 * n, 2 * n)), J_HHL], [J_HHL.T, np.zeros((2 * n, 2 * n))]])  # 2n*2n次元の行列
        b_HHL = np.block([b_HHL, np.zeros(2 * n)])  # 2n次元のベクトル
        # Wの固有値を確認 -> [-pi, pi] に収まっている
        for value in np.linalg.eigh(J_HHL)[0]:
            if value < -np.pi or np.pi < value:
                print("caution:eigenvalues exist that are not included in [-π,π].")
                exit(1)

        y_HHL = HHL_algorithm(J_HHL, b_HHL, dim, reg_nbit, scale_fac)
        print(np.average(np.abs(y_HHL[2 * n:3 * n])))

        HHL_first += y_HHL[2 * n:3 * n]  # 解の更新
        diff_HHL_first += y_HHL[3 * n:4 * n]

    t = time.time() - start
    HHL_runge_result[index] = HHL_first

    with open("2階ODE数値結果/HHL(number=" + str(number) + ",reg_qubits=" + str(reg_nbit) + ",h=" + str(h) + ",n=" + str(
            n) + ").csv", 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(n):
            writer.writerow([x_list[i], HHL_runge_result[index][i]])

    print("HHL calculation completed.")
    print("Elapsed time[s]:", t)
    print("register qubits:", reg_nbit)
    print("number:", number, "h:", h, " n:", n, "\n")

###################### Runge-kutta法(4次・HHL)専用 ######################
del HHL_runge_result
