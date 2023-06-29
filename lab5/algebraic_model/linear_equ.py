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
    if i == j:
        if i != 6:
            for number in range(6):
                d_ij[i+number*6] = 1
        if i == 6:
            for number in range(6):
                d_ij[i-1+number*6] = 1
    else:
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
    return d_ij*40


def get_length_RS(i, j, n):
    # y = (j-i)/6*x+i ==>  x=(6y-6i)/(j-i)
    # 直线方程： x = col_id ; y = row_id
    x = np.zeros(n*n)
    d_ij = np.zeros(n*n)
    if i == j:
        if i != 6:
            for number in range(6):
                d_ij[i+number*6] = 1
        if i == 6:
            for number in range(6):
                d_ij[i-1+number*6] = 1
    else:
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
    return d_ij*40


def get_matrix_D(n, P, Q, R, S):
    D_init = np.array([np.zeros(n*n)])
    for i in range(7):
        # for index_P in range(len(P)):
        #     if i != P[index_P]:
        if i not in P:
            for j in range(7):
                # for index_Q in range(len(Q)):
                #     if index_Q != j:
                if j not in Q:
                    d_ij = get_length_PQ(i, j, n)
                    D_init = np.vstack((D_init, d_ij.reshape(1, n*n)))
    for k in range(7):
        # for index_R in range(len(R)):
        #     if k != index_R:
        if k not in R:
            for l in range(7):
                # for index_S in range(len(S)):
                #     if index_S != l:
                if l not in S:
                    d_ij = get_length_RS(k, l, n)
                    D_init = np.vstack((D_init, d_ij.reshape(1, n*n)))
    D = np.delete(D_init, 0, axis=0)
    return D


def get_vector_t(P, Q, R, S):  # 传入参数为要删去的数据
    T_PQ = np.array([[0.0611, 0.0895, 0.1996, 0.2032, 0.4181, 0.4923, 0.5646],
                  [0.0989, 0.0592, 0.4413, 0.4318, 0.4770, 0.5242, 0.3805],
                  [0.3052, 0.4131, 0.0598, 0.4153, 0.4156, 0.3563, 0.1919],
                  [0.3221, 0.4453, 0.4040, 0.0738, 0.1789, 0.0740, 0.2122],
                  [0.3490, 0.4529, 0.2263, 0.1917, 0.0839, 0.1768, 0.1810],
                  [0.3807, 0.3177, 0.2364, 0.3064, 0.2217, 0.0939, 0.1031],
                  [0.4311, 0.3397, 0.3566, 0.1954, 0.0760, 0.0688, 0.1042]])

    T_RS = np.array([[0.0645, 0.0602, 0.0813, 0.3516, 0.3867, 0.4314, 0.5721],
                  [0.0753, 0.0700, 0.2852, 0.4341, 0.3491, 0.4800, 0.4980],
                  [0.3456, 0.3205, 0.0974, 0.4093, 0.4240, 0.4540, 0.3112],
                  [0.3655, 0.3289, 0.4247, 0.1007, 0.3249, 0.2134, 0.1017],
                  [0.3165, 0.2409, 0.3214, 0.3256, 0.0904, 0.1874, 0.2130],
                  [0.2749, 0.3891, 0.5895, 0.3016, 0.2058, 0.0841, 0.0706],
                  [0.4434, 0.4919, 0.3904, 0.0786, 0.0709, 0.0914, 0.0583]])
    # for i in range(len(P)):
    T_PQ = np.delete(T_PQ, P, axis=0)
    # for i in range(len(Q)):
    T_PQ = np.delete(T_PQ, Q, axis=1)
    # for i in range(len(R)):
    T_RS = np.delete(T_RS, R, axis=0)
    # for i in range(len(S)):
    T_RS = np.delete(T_RS, S, axis=1)
    t1 = T_PQ.flatten()
    t2 = T_RS.flatten()
    t = np.concatenate((t1, t2))
    return t


def get_x(n, P, Q, R, S):
    D = get_matrix_D(n, P, Q, R, S)
    t = get_vector_t(P, Q, R, S)
    vector_one = np.ones(n*n)
    # print(D.shape)
    # print(t.shape)
    t_ = t - np.dot(D, vector_one)/320/40
    A = (1/2880 - 1/320) * D
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
    # print('%d*%d划分中删去指定的波源和接收器后所得空洞位置的近似解为(1表示该位置有空洞)：' % (n, n))
    # print(x1)
    exect_res = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0,
                          0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    error = np.linalg.norm(exect_res-x1)
    print('与使用全部数据的误差为：%f\n' % error)
    return error




if __name__ == '__main__':
    # 传入的P, Q, R, S参数表示要删去的波源或接收器的编号（从0开始）
    # P, Q, R, S = [], [], [], [0, 1, 2]
    # err = []
    # for u in range(7):
    #     S[0] = u
    #     for v in range(u+1, 7):
    #         S[1] = v
    #         for w in range(v+1, 7):
    #             S[2] = w
    #             erroring = get_x(6, P, Q, R, S)
    #             err.append(erroring)
    # print(err)
    P, Q, R, S = [5], [], [], []
    get_x(6, P, Q, R, S)
