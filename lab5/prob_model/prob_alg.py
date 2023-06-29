import d_ij
import numpy as np
from scipy.special import softmax



def get_line_red_all():
    """计算直线上空洞的比例"""
    m=6
    t1 = np.array([[0.0611, 0.0895, 0.1996, 0.2032, 0.4181, 0.4923, 0.5646],
       [0.0989, 0.0592, 0.4413, 0.4318, 0.477 , 0.5242, 0.3805],
       [0.3052, 0.4131, 0.0598, 0.4153, 0.4156, 0.3563, 0.1919],
       [0.3221, 0.4453, 0.404 , 0.0738, 0.1789, 0.074 , 0.2122],
       [0.349 , 0.4529, 0.2263, 0.1917, 0.0839, 0.1768, 0.181 ],
       [0.3807, 0.3177, 0.2364, 0.3064, 0.2217, 0.0939, 0.1031],
       [0.4311, 0.3397, 0.3566, 0.1954, 0.076 , 0.0688, 0.1042]])
    t2 = np.array([[0.0645, 0.0602, 0.0813, 0.3516, 0.3867, 0.4314, 0.5721],
       [0.0753, 0.07  , 0.2852, 0.4341, 0.3491, 0.48  , 0.498 ],
       [0.3456, 0.3205, 0.0974, 0.4093, 0.424 , 0.454 , 0.3112],
       [0.3655, 0.3289, 0.4247, 0.1007, 0.3249, 0.2134, 0.1017],
       [0.3165, 0.2409, 0.3214, 0.3256, 0.0904, 0.1874, 0.213 ],
       [0.2749, 0.3891, 0.5895, 0.3016, 0.2058, 0.0841, 0.0706],
       [0.4434, 0.4919, 0.3904, 0.0786, 0.0709, 0.0914, 0.0583]])
    p_line = np.zeros((m, m, 2))
    for i in range(m):
        for j in range(m):
            d_ij= (m**2 + (i-j)**2)**0.5*40
            p_line[i, j, 0] = max(
                (2880 * 320 * t1[i, j] - d_ij * 320) / (d_ij * (2880 - 320)), 0)
            p_line[i, j, 1] = max(
                (2880 * 320 * t2[i, j] - d_ij * 320) / (d_ij * (2880 - 320)), 0)

    return p_line


def get_line_red():
    """计算直线上空洞的比例"""
    m = 6
    t1 = np.array([[0.0611, 0.0895, 0.1996, 0.2032, 0.4181, 0.4923, 0.5646],
       [0.0989, 0.0592, 0.4413, 0.4318, 0.477 , 0.5242, 0.3805],
       [0.3052, 0.4131, 0.0598, 0.4153, 0.4156, 0.3563, 0.1919],
       [0.3221, 0.4453, 0.404 , 0.0738, 0.1789, 0.074 , 0.2122],
       [0.349 , 0.4529, 0.2263, 0.1917, 0.0839, 0.1768, 0.181 ],
       [0.3807, 0.3177, 0.2364, 0.3064, 0.2217, 0.0939, 0.1031],
       [0.4311, 0.3397, 0.3566, 0.1954, 0.076 , 0.0688, 0.1042]])
    p_line = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            d_ij= (m**2 + (i-j)**2)**0.5*40
            p_line[i, j] = max(
                (2880 * 320 * t1[i, j] - d_ij * 320) / (d_ij * (2880 - 320)), 0)

    return p_line

def get_dis(m):
    """计算接收器之间的距离"""
    dis = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            dis[i, j] = (m**2 + (i-j)**2)**0.5 * 40

    return dis

def get_length(m = 6):
    """计算直线 (i, j) 在第 n 个小正方形内的长度"""
    d = np.zeros((m, m, m**2))  # 直线 (i, j) 在第 n 个小正方形内的长度
    for i in range(m):
        for j in range(m):
            if i != j:
                d[i, j] = d_ij.get_length_PQ(i, j, m)

    return d

def get_P(m=6, b=6):
    """计算小正方形 n 为空洞的概率
    + m: 小正方形个数
    + b: 需要的发射器个数 - 1
    """
    P = np.zeros(m**2)  # 小正方形 n 为空洞的概率
    d = get_length(m)  # 直线 (i, j) 在第 n 个小正方形内的长度
    P_2 = np.zeros(m**2)
    p_line = get_line_red_all()  # 直线 (i, j) 上空洞的比例
    if m > 6 or b < 6: # 假设 m % b == 0, 且 6 // b == 0
        stride = m // b
        d = d[::stride, ::stride]  # [b, b, m**2]
        stride = 6 // b
        p_line = p_line[::stride, ::stride]  # [b, b, 2]
    # 计算小正方形 n 为空洞的概率
    for n in range(m**2):
        P[n] = (p_line[...,0] * d[...,n]).sum() / (d[...,n].sum() + 1e-10)
        P_2[n] = (p_line[...,1] * d[...,n]).sum() / (d[...,n].sum() + 1e-10)
    P_2 = P_2.reshape((m, m)).T.flatten()
    P = (P + P_2) / 2
    return P


def get_P_iter(m=6, b=6, epsilon=1e-3, iter_num=100):
    """迭代算法计算小正方形 n 为空洞的概率
    + m: 小正方形个数
    + b: 需要的发射器个数 - 1
    """
    d = get_length(m)  # 直线 (i, j) 在第 n 个小正方形内的长度
    P = get_P(m, b)
    P_2 = np.zeros(m**2)
    P_new = np.zeros(m**2)
    p_line = get_line_red_all().reshape(6, 6, 1, 2).repeat(m**2, axis=2) 
    p_line_new = np.zeros_like(p_line)
    if m > 6 or b < 6: # 假设 m % b == 0, 且 6 // b == 0
        stride = m // b
        d = d[::stride, ::stride]  # [b, b, m**2]
        stride = 6 // b
        p_line = p_line[::stride, ::stride]  # [b, b, 2]
        p_line_new = p_line_new[::stride, ::stride]  # [b, b, 2]
    for i in range(iter_num):
        if abs(P_new - P).sum() < epsilon:
            return i, get_P(m, b) , P
        P_new = P.copy()
        p_line_new = p_line.copy()
        alpha = softmax(P.reshape(1,1,-1) * d, 2)  # 加权和 shape==(b, b, m**2)
        p_line[...,0] = alpha * p_line_new[...,0]  # \sum alpha_i * x_i
        alpha = np.transpose(alpha.reshape(b,b,m,m), (0,1,3,2)).reshape(b,b,m**2)
        p_line[...,1] = alpha * p_line_new[...,1]
        for n in range(m**2):
            P[n] = (p_line[...,n, 0] * d[...,n]).sum() / (d[...,n].sum() + 1e-10)
            P_2[n] = (p_line[...,n, 1] * d[...,n]).sum() / (d[...,n].sum() + 1e-10)
        P_2 = P_2.reshape((m, m)).T.flatten()
        P = (P + P_2) / 2

    return iter_num, get_P(m, b), P

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    print(get_P_iter(6, 1e-3))