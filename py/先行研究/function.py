from import_summary import *
from utility import *  # 計算基底に関する確率分布を表示
from HHL_function import *  # 計算基底に関する確率分布を表示

# 設定
np.set_printoptions(linewidth=200)
# dy/dx = f(x)y + g(x)
# e.g)dy/dy = -2xy + 2x
# 各パラメータ
number = 2
h = 0.05  # 刻み幅
n = 200  # 考える区間[x_first, x_first + h*(n-1)]
scale_fac = 1.0
reg_nbit = 12  # 位相推定に使うレジスタの数
x_list = np.array([h * i for i in range(n)])  # 区間[0,h(n-1)]と厳密解
if number == 1:
    y_exact = np.exp(-x_list ** 2) + 1
    y_first = 2  # 初期値y(0)


    def f_function(x):
        return -2 * x


    def g_function(x):
        return 2 * x

elif number == 2:
    y_exact = np.exp(-2 * x_list) + 1
    y_first = 2  # 初期値y(0)


    def f_function(x):
        return -2


    def g_function(x):
        return 2