import numpy as np
# 将网络划分为n*n个小正方形
# 暴力枚举出直线在每个小正方形的内的长度


def compute_dis(x_1, y_1, x_2, y_2):
    if x_1 is None or x_2 is None or y_1 is None or y_2 is None:
        return 0
    else:
        return ((x_2-x_1)**2+(y_2-y_1)**2)**0.5


def get_length_PQ(i, j, n):
    # y = 6/(i-j)*(x-j) ==> (i-j)y=6(x-j) ==> 6x-(i-j)y = 6j
    # 直线方程： x = col_id ; y = row_id
    # x = np.zeros(36)
    # d_ij = np.zeros(36)
    x = np.zeros(n*n)
    d_ij = np.zeros(n*n)
    for num in range(n*n):
        # 每个小正方形用左上角的点表示
        row_id = num // n
        col_id = num % n
        x_1, y_1, x_2, y_2 = None, None, None, None
        x_3, y_3, x_4, y_4 = None, None, None, None
        y_temp_1 = n / (i-j) * (col_id-j)
        x_temp_1 = (n*j+(i-j)*row_id)/n
        y_temp_2 = n / (i - j) * (col_id+1-j)
        x_temp_2 = (n * j + (i - j) * (row_id+1)) / n
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
    # 计算P_iQ_j的长度和通过介质的长度
    q_ij, p_ij = 0, 0
    for k in range(n*n):
        q_ij += 40 * d_ij[k] * x[k]  # 通过空气的总长度
        p_ij += 40 * d_ij[k] * (1-x[k])  # 通过介质的总长度
    return d_ij * 240 / n


def get_length_RS(i, j, n):
    # y = (j-i)/6*x+i ==>  x=(6y-6i)/(j-i)
    # 直线方程： x = col_id ; y = row_id
    x = np.zeros(n*n)
    d_ij = np.zeros(n*n)
    for num in range(n*n):
        row_id = num // n
        col_id = num % n
        x_1, y_1, x_2, y_2 = None, None, None, None
        x_3, y_3, x_4, y_4 = None, None, None, None
        y_temp_1 = (j-i)/n*col_id + i
        x_temp_1 = (n*row_id-n*i)/(j-i)
        y_temp_2 = (j-i)/n*(col_id+1) + i
        x_temp_2 = (n*(row_id+1)-n*i)/(j-i)
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
    # 计算P_iQ_j的长度和通过介质的长度
    q_ij, p_ij = 0, 0
    for k in range(n*n):
        q_ij += 40 * d_ij[k] * x[k]  # 通过空气的总长度
        p_ij += 40 * d_ij[k] * (1-x[k])  # 通过介质的总长度
    return d_ij*240/n


def get_matrix_D(n):
    D_init = np.array([np.zeros(n*n)])
    for i in range(7):
        for j in range(7):
            if j != i:
                d_ij = get_length_PQ(i, j, n)
                D_init = np.vstack((D_init, d_ij.reshape(1, n*n)))
    for k in range(7):
        for l in range(7):
            if k != l:
                d_ij = get_length_RS(k, l, n)
                D_init = np.vstack((D_init, d_ij.reshape(1, n*n)))
    D = np.delete(D_init, 0, axis=0)
    return D


def get_vector_t():
    # t = np.array([0.0611, 0.0895, 0.1996, 0.2032, 0.4181, 0.4923, 0.5646,
    #               0.0989, 0.0592, 0.4413, 0.4318, 0.4770, 0.5242, 0.3805,
    #               0.3052, 0.4131, 0.0598, 0.4153, 0.4156, 0.3563, 0.1919,
    #               0.3221, 0.4453, 0.4040, 0.0738, 0.1789, 0.0740, 0.2122,
    #               0.3490, 0.4529, 0.2263, 0.1917, 0.0839, 0.1768, 0.1810,
    #               0.3807, 0.3177, 0.2364, 0.3064, 0.2217, 0.0939, 0.1031,
    #               0.4311, 0.3397, 0.3566, 0.1954, 0.0760, 0.0688, 0.1042,
    #
    #               0.0645, 0.0602, 0.0813, 0.3516, 0.3867, 0.4314, 0.5721,
    #               0.0753, 0.0700, 0.2852, 0.4341, 0.3491, 0.4800, 0.4980,
    #               0.3456, 0.3205, 0.0974, 0.4093, 0.4240, 0.4540, 0.3112,
    #               0.3655, 0.3289, 0.4247, 0.1007, 0.3249, 0.2134, 0.1017,
    #               0.3165, 0.2409, 0.3214, 0.3256, 0.0904, 0.1874, 0.2130,
    #               0.2749, 0.3891, 0.5895, 0.3016, 0.2058, 0.0841, 0.0706,
    #               0.4434, 0.4919, 0.3904, 0.0786, 0.0709, 0.0914, 0.0583])

    t = np.array([0.0895, 0.1996, 0.2032, 0.4181, 0.4923, 0.5646,
                  0.0989, 0.4413, 0.4318, 0.4770, 0.5242, 0.3805,
                  0.3052, 0.4131, 0.4153, 0.4156, 0.3563, 0.1919,
                  0.3221, 0.4453, 0.4040, 0.1789, 0.0740, 0.2122,
                  0.3490, 0.4529, 0.2263, 0.1917, 0.1768, 0.1810,
                  0.3807, 0.3177, 0.2364, 0.3064, 0.2217, 0.1031,
                  0.4311, 0.3397, 0.3566, 0.1954, 0.0760, 0.0688,

                  0.0602, 0.0813, 0.3516, 0.3867, 0.4314, 0.5721,
                  0.0753, 0.2852, 0.4341, 0.3491, 0.4800, 0.4980,
                  0.3456, 0.3205, 0.4093, 0.4240, 0.4540, 0.3112,
                  0.3655, 0.3289, 0.4247, 0.3249, 0.2134, 0.1017,
                  0.3165, 0.2409, 0.3214, 0.3256, 0.1874, 0.2130,
                  0.2749, 0.3891, 0.5895, 0.3016, 0.2058, 0.0706,
                  0.4434, 0.4919, 0.3904, 0.0786, 0.0709, 0.0914])
    return t


def get_x(n):
    D = get_matrix_D(n)
    t = get_vector_t()
    vector_one = np.ones(n*n)
    # print(D[0])
    # print(np.dot(D, vector_one)/320)
    t_ = t - np.dot(D, vector_one)/320
    print(t_)
    A = (1/2880 - 1/320) * D
    print(A[-1])
    # 最小二乘解出x
    x1 = np.linalg.lstsq(A, t_, rcond=None)[0]
    # temp = np.linalg.inv(np.dot(A.T, A))
    # x = np.dot(np.dot(temp, A.T), t_)
    print('最小二乘解为：')
    print(x1)
    for i in range(n*n):
        # if abs(abs(x1[i])-1) <= 0.26:
        if abs(x1[i]) >= 0.48*(max(abs(x1))-min(abs(x1))):
            x1[i] = 1
        else:
            x1[i] = 0
    print('%d*%d划分中所得空洞位置的近似解为(1表示该位置有空洞)：'% (n,n))
    print(x1)
    error = np.linalg.norm(np.dot(A, x1)-t_)
    print('误差向量长度平方为：%f\n' % error)




if __name__ == '__main__':
    # 传入参数为划分网格的个数（参数一般设置为6，12等）
    get_x(12)
