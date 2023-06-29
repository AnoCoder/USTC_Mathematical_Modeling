import numpy as np
# 利用贪心法对空洞探测问题的数据进行分析后设定合理的误差
# 以理论时间与实际时间的误差作为贪心选择标准确定空洞所处的区域以及空洞的个数．算法复杂度为多项式级


def compute_dis(x_1, y_1, x_2, y_2):
    if x_1 is None or x_2 is None or y_1 is None or y_2 is None:
        return 0
    else:
        return ((x_2-x_1)**2+(y_2-y_1)**2)**0.5


# 计算波沿直线P_iQ_j传播的时间
def cal_t_PQ(i, j, loc):  # loc表示空洞的坐标
    d_ij = np.zeros(36)
    for num in range(36):
        # 每个小正方形用左上角的点表示
        row_id = num // 6
        col_id = num % 6
        x_1, y_1, x_2, y_2 = None, None, None, None
        x_3, y_3, x_4, y_4 = None, None, None, None
        y_temp_1 = 6 / (i - j) * (col_id - j)
        x_temp_1 = (6 * j + (i - j) * row_id) / 6
        y_temp_2 = 6 / (i - j) * (col_id + 1 - j)
        x_temp_2 = (6 * j + (i - j) * (row_id + 1)) / 6
        if row_id <= y_temp_1 <= row_id + 1:
            x_1 = col_id
            y_1 = y_temp_1
        if col_id <= x_temp_1 <= col_id + 1:
            x_2 = x_temp_1
            y_2 = row_id
        if row_id <= y_temp_2 <= row_id + 1:
            x_3 = col_id + 1
            y_3 = y_temp_2
        if col_id <= x_temp_2 <= col_id + 1:
            x_4 = x_temp_2
            y_4 = row_id + 1
        dis = max(compute_dis(x_1, y_1, x_2, y_2), compute_dis(x_1, y_1, x_3, y_3),
                  compute_dis(x_1, y_1, x_4, y_4), compute_dis(x_2, y_2, x_3, y_3),
                  compute_dis(x_2, y_2, x_4, y_4), compute_dis(x_3, y_3, x_4, y_4))
        if dis != 0:
            d_ij[num] = dis * 40
    t_ij = 0.0

    for k in range(36):
        modify = False
        for l in range(len(loc)):
            if k == loc[l][0]*6+loc[l][1]:
                modify = True
                t_ij += d_ij[k]/320
        if not modify:
            t_ij += d_ij[k]/2880
    return t_ij


# 计算波沿直线R_iS_j传播的时间
def cal_t_RS(i, j, loc):  # a, b表示空洞的坐标
    d_ij = np.zeros(36)
    for num in range(36):
        # 每个小正方形用左上角的点表示
        row_id = num // 6
        col_id = num % 6
        x_1, y_1, x_2, y_2 = None, None, None, None
        x_3, y_3, x_4, y_4 = None, None, None, None
        y_temp_1 = (j - i) / 6 * col_id + i
        x_temp_1 = (6 * row_id - 6 * i) / (j - i)
        y_temp_2 = (j - i) / 6 * (col_id + 1) + i
        x_temp_2 = (6 * (row_id + 1) - 6 * i) / (j - i)
        if row_id <= y_temp_1 <= row_id + 1:
            x_1 = col_id
            y_1 = y_temp_1
        if col_id <= x_temp_1 <= col_id + 1:
            x_2 = x_temp_1
            y_2 = row_id
        if row_id <= y_temp_2 <= row_id + 1:
            x_3 = col_id + 1
            y_3 = y_temp_2
        if col_id <= x_temp_2 <= col_id + 1:
            x_4 = x_temp_2
            y_4 = row_id + 1
        dis = max(compute_dis(x_1, y_1, x_2, y_2), compute_dis(x_1, y_1, x_3, y_3),
                  compute_dis(x_1, y_1, x_4, y_4), compute_dis(x_2, y_2, x_3, y_3),
                  compute_dis(x_2, y_2, x_4, y_4), compute_dis(x_3, y_3, x_4, y_4))
        if dis != 0:
            d_ij[num] = dis * 40
    tao_ij = 0.0
    for k in range(36):
        modify = False
        for l in range(len(loc)):
            if k == loc[l][0] * 6 + loc[l][1]:
                modify = True
                tao_ij += d_ij[k] / 320
        if not modify:
            tao_ij += d_ij[k] / 2880
    return tao_ij


def get_delta_t(loc):
    t = np.array([0.0611, 0.0895, 0.1996, 0.2032, 0.4181, 0.4923, 0.5646,
                  0.0989, 0.0592, 0.4413, 0.4318, 0.4770, 0.5242, 0.3805,
                  0.3052, 0.4131, 0.0598, 0.4153, 0.4156, 0.3563, 0.1919,
                  0.3221, 0.4453, 0.4040, 0.0738, 0.1789, 0.0740, 0.2122,
                  0.3490, 0.4529, 0.2263, 0.1917, 0.0839, 0.1768, 0.1810,
                  0.3807, 0.3177, 0.2364, 0.3064, 0.2217, 0.0939, 0.1031,
                  0.4311, 0.3397, 0.3566, 0.1954, 0.0760, 0.0688, 0.1042,

                  0.0645, 0.0602, 0.0813, 0.3516, 0.3867, 0.4314, 0.5721,
                  0.0753, 0.0700, 0.2852, 0.4341, 0.3491, 0.4800, 0.4980,
                  0.3456, 0.3205, 0.0974, 0.4093, 0.4240, 0.4540, 0.3112,
                  0.3655, 0.3289, 0.4247, 0.1007, 0.3249, 0.2134, 0.1017,
                  0.3165, 0.2409, 0.3214, 0.3256, 0.0904, 0.1874, 0.2130,
                  0.2749, 0.3891, 0.5895, 0.3016, 0.2058, 0.0841, 0.0706,
                  0.4434, 0.4919, 0.3904, 0.0786, 0.0709, 0.0914, 0.0583])
    T = np.zeros((7, 7))
    Tao = np.zeros((7, 7))
    for i in range(7):  # 7个探测器，7个接收器
        for j in range(7):
            if i != j:
                T[i][j] = cal_t_PQ(i, j, loc)
                Tao[i][j] = cal_t_RS(i, j, loc)
            else:
                T[i][j] = t[i*7+j]
                Tao[i][j] = t[49+i*7+j]
    delta_t1 = np.linalg.norm(T.reshape(-1)-t[:49])
    delta_t2 = np.linalg.norm(Tao.reshape(-1)-t[49:])
    delta_t = delta_t1 + delta_t2
    # print(delta_t1, delta_t2)
    return delta_t


if __name__ == '__main__':
    delta_t_min = 2.3266+2.7728  # 没有空洞时计算的理论传播时间与实际测量时间的误差
    # 假设有一个空洞（即先确认出第一个空洞的位置）
    for m in range(6):
        for n in range(6):
            temp_t = get_delta_t([[m, n]])
            if temp_t <= delta_t_min:
                delta_t_min = temp_t
                loc_1 = [m, n]
    print('贪心法得到第1个空洞位置：\t%s' % loc_1)
    print('此时理论时间与实际时间的误差为：\t%f' % delta_t_min)
    loc = np.array(loc_1)
    for num in range(1, 36):
        t_record = []
        for m in range(6):
            for n in range(6):
                temp_t = get_delta_t(np.vstack((loc, [m, n])))
                t_record.append(temp_t)
        if min(t_record) > delta_t_min:
            print('此时算得的误差增大，空洞数目不能再增加，贪心法结束')
            print('贪心法所得所有空洞位置为：')
            for k in range(loc.shape[0]):
                print('%d:\t%s' % (k+1, loc[k]))
            print('算法终止时的误差：\t%f' % delta_t_min)
            break
        else:
            delta_t_min = min(t_record)
            min_index = np.argmin(t_record)
            loc = np.vstack((loc, [min_index//6, min_index % 6]))
            print('贪心法得到第 %d 个空洞位置：\t%s' % (num+1, [min_index//6, min_index % 6]))
            print('理论时间与实际时间的误差为：\t%f' % delta_t_min)




