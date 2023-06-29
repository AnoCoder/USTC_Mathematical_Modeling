import numpy as np


# 将网络划分为36个小正方形
# 暴力枚举出直线在每个小正方形的内的长度

def compute_dis(x_1, y_1, x_2, y_2):
    if x_1 is None or x_2 is None or y_1 is None or y_2 is None:
        return 0
    else:
        return ((x_2-x_1)**2+(y_2-y_1)**2)**0.5


def get_length_PQ(i, j, m=6):
    # y = 6/(i-j)*(x-j) ==> (i-j)y=6(x-j) ==> 6x-(i-j)y = 6j
    # 直线方程： x = col_id ; y = row_id
    x = np.zeros(m**2)
    d_ij = np.zeros(m**2)
    for num in range(m**2):
        # 每个小正方形用左上角的点表示
        row_id = num // m
        col_id = num % m
        x_1, y_1, x_2, y_2 = None, None, None, None
        x_3, y_3, x_4, y_4 = None, None, None, None
        y_temp_1 = m / (i-j) * (col_id-j)
        x_temp_1 = (m * j + (i - j) * row_id)/m
        y_temp_2 = m / (i - j) * (col_id+1-j)
        x_temp_2 = (m * j + (i - j) * (row_id+1)) / m
        if row_id <= y_temp_1 <= row_id+1:
            x_1 = col_id
            y_1 = y_temp_1
        if col_id <= x_temp_1 <= col_id+1:
            x_2 = x_temp_1
            y_2 = row_id
        if row_id <= y_temp_2 <= row_id+1:
            x_3 = col_id+1
            y_3 = y_temp_2
        if col_id <= x_temp_2 <= col_id+1:
            x_4 = x_temp_2
            y_4 = row_id+1
        dis = max(compute_dis(x_1, y_1, x_2, y_2), compute_dis(x_1, y_1, x_3, y_3),
                  compute_dis(x_1, y_1, x_4, y_4), compute_dis(x_2, y_2, x_3, y_3),
                  compute_dis(x_2, y_2, x_4, y_4), compute_dis(x_3, y_3, x_4, y_4))
        if dis != 0:
            d_ij[num] = dis
            x[num] = 1
    return d_ij


if __name__ == '__main__':
    d_ij = get_length_PQ(2, 5)
    print(d_ij)
