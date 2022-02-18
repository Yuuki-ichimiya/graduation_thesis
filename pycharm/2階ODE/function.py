from import_summary import *
from utility import *  # 計算基底に関する確率分布を表示
from HHL_function import *  # HHL algorithmに必要な関数
from copy import copy
import tqdm

np.set_printoptions(linewidth=200, precision=20)

number = 6
if number == 1:
    h = 0.01  # 刻み幅
    n = 100  # 考える区間[x_first, x_first + h*(n-1)]
    y_order = 1
    z_order = 1
    x_list = np.array([h * i for i in range(n)])
    y_exact = 2 / 5 * np.exp(5 * x_list) + np.exp(-x_list) - x_list - 2 / 5  # y(0)=1, dy/dx(0)=0
    diff_y_exact = 2 * np.exp(5 * x_list) - np.exp(-x_list) - 1
    y_first = copy(y_exact)
    diff_y_first = copy(diff_y_exact)

elif number == 2:
    # y = sin(x) + 2
    # d^2y/dx^2 = - y + 2, y(x_0) = 2, dy/dx(x_0) = 1
    h = 0.064  # 刻み幅
    n = 100  # 考える区間[x_first, x_first + h*(n-1)]
    y_order = 0
    z_order = 1
    x_list = np.array([h * i for i in range(n)])
    y_exact = np.sin(x_list) + 2  # y(0)=1, dy/dx(0)=0
    diff_y_exact = np.cos(x_list)
    y_first = copy(y_exact)
    diff_y_first = copy(diff_y_exact)

elif number == 3:
    # y = 2(exp(-x/4)sin(4x) + 1)
    # d^2y/dx^2=-1/2*dy/dx - 257y/16 + 257/8,y(0)=2,y'(0)=8
    h = 0.064  # 刻み幅
    n = 100  # 考える区間[x_first, x_first + h*(n-1)]
    y_order = 1
    z_order = 1
    x_list = np.array([h * i for i in range(n)])
    y_exact = 2 * (np.exp(-x_list / 4) * np.sin(4 * x_list) + 1)  # y(0)=1, dy/dx(0)=0
    diff_y_exact = 2 * (
            -1 / 4 * np.exp(-x_list / 4) * np.sin(4 * x_list) + 4 * np.exp(-x_list / 4) * np.cos(4 * x_list))
    y_first = copy(y_exact)
    diff_y_first = copy(diff_y_exact)


elif number == 4:
    # d^2y/dt^2=b(dy/dt)-a(dy/dt)^2-s_cy,y(0)=0, y'(0)=0
    a_para = 250
    b_para = 50
    s_c_para = 1.2
    c_para = 1

    h = 0.01  # 刻み幅
    n = 100  # 考える区間[x_first, x_first + h*(n-1)]
    y_order = 1
    z_order = 2
    x_list = np.array([h * i for i in range(n)])
    y_exact = np.array([0.001])  # y(0)=3,y'(0)=7
    diff_y_exact = np.array([0])
    y_first = copy(y_exact)
    diff_y_first = copy(diff_y_exact)


elif number == 5:
    # d^2y/dx^2 = - g/l * sin(y), y(x_0) = 2, dy/dx(x_0) = 1
    omega = np.sqrt(9.80665 / 10)
    theta_max = np.pi / 3
    k_para = np.sin(theta_max / 2)

    h = 0.05  # 刻み幅
    n = 150  # 考える区間[x_first, x_first + h*(n-1)]
    y_order = 5
    z_order = 1
    x_list = np.array([h * i for i in range(n)])
    y_first = np.array([0])
    diff_y_first = np.array([2 * omega * k_para])

    # beta = np.arctan(omega*y_first[0]/diff_y_first[0])
    # alpha = y_first[0]/np.sin(beta)
    # y_exact = alpha * np.sin(omega*x_list + beta)

    y_exact = np.array([])
    for i in range(x_list.shape[0]):
        y_exact = np.append(y_exact, 2 * np.arcsin(k_para * special.ellipj(omega * x_list[i], k_para ** 2)[0]))


elif number == 6:
    # d^2y/dt^2=-μ(y^2-1)(dy/dt)-y,y(0)=0, y'(0)=0
    mu = 0.7

    h = 0.01  # 刻み幅
    n = 300  # 考える区間[x_first, x_first + h*(n-1)]
    y_order = 2
    z_order = 1
    x_list = np.array([h * i for i in range(n)])

    y_0 = 0.5  # y(0)
    temp_func = (4 / y_0 ** 2 - 1) * np.exp(-mu * x_list)  # (4/a^2_0-1)exp(-μt)*cos(t)
    y_exact = 2 / np.sqrt(1 + temp_func) * np.cos(x_list)  # y(0)=0.5,y'(0)=0
    diff_y_exact = -2 / np.sqrt(1 + temp_func) * np.sin(x_list)
    # diff_y_exact = mu*temp_func/((1+temp_func)*np.sqrt(1+temp_func))*np.cos(x_list)-2/np.sqrt(1+temp_func)*np.sin(
    # x_list)

    y_first = copy(y_exact)
    diff_y_first = copy(diff_y_exact)

else:
    print("Invalid number.")
    sys.exit(1)


def function(x, axis):
    # dy/dx = zの右辺
    func = np.zeros((1 + y_order, 1 + z_order))
    if number == 1:
        if axis == "y":
            func[0][1] = 1
            return func

        # dz/dx = 4z + 5y + (5x+6)
        elif axis == "z":
            func = np.zeros((1 + y_order, 1 + z_order))
            func[0][0] = 5 * x + 6
            func[1][0] = 5
            func[0][1] = 4
            return func

        else:
            print("unknown axis")
            sys.exit(1)

    elif number == 2:
        if axis == "y":
            func[0][1] = 1
            return func

        # dz/dx = -y + 2
        elif axis == "z":
            func = np.zeros((1 + y_order, 1 + z_order))
            func[0][0] = 2
            func[1][0] = -1
            return func

        else:
            print("unknown axis")
            sys.exit(1)

    elif number == 3:
        if axis == "y":
            func[0][1] = 1
            return func

        # dz/dx=-1/2*z - 257y/16 + 257/8
        elif axis == "z":
            func = np.zeros((1 + y_order, 1 + z_order))
            func[0][0] = 257 / 8
            func[1][0] = -257 / 16
            func[0][1] = -1 / 2
            return func

        else:
            print("unknown axis")
            sys.exit(1)

    elif number == 4:
        if axis == "y":
            func[0][1] = 1
            return func

        # dz/dt=bz-az^2-s_cy
        elif axis == "z":
            func = np.zeros((1 + y_order, 1 + z_order))
            func[0][1] = b_para
            func[0][2] = -a_para
            func[1][0] = -s_c_para

            return func

        else:
            print("unknown axis")
            sys.exit(1)

    elif number == 5:
        g = 9.80665
        l = 10
        # dy/dx = z
        if axis == "y":
            func[0][1] = 1
            return func

        # dz/dx = -g/l sin(y)≒-g/l(y-y^3/3!+y^5/5!+...)
        elif axis == "z":
            func = np.zeros((1 + y_order, 1 + z_order))
            func[1][0] = -g / l * (1 / math.factorial(1))
            func[3][0] = g / l * (1 / math.factorial(3))
            func[5][0] = -g / l * (1 / math.factorial(5))

            return func

        else:
            print("unknown axis")
            sys.exit(1)

    elif number == 6:
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


# 各パラメータ
scale_fac = 1.0
reg_nbit_list = np.array([9, 10, 11])  # 位相推定に使うレジスタの数(9,10,11)
depth = int(np.floor(np.log2(n)))

max_ydim = 2 * y_order ** 2 * z_order + y_order ** 2 + y_order * z_order ** 2 * (z_order + 1)  # kはy_iの多項式の次数
max_zdim = 3 * y_order * z_order ** 2 + y_order ** 2 + z_order ** 4  # kはz_iの多項式の次数
scale_fac = 1.0
reg_nbit_list = np.array([9, 10, 11])  # 位相推定に使うレジスタの数(9,10,11)
