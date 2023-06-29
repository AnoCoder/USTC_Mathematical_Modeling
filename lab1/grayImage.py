import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from functools import partial


""""
运行代码时生成一个图片后需要将该图片关闭后才能接着运行
"""


def show_image(image):
    plt.imshow(image.astype('uint8'), interpolation="none")
    plt.axis('off')  # 去掉坐标轴
    plt.show()


# 加权产生灰度图的函数，用于辅助 class类里面 max_entropy()函数产生66个备选灰度图
def convert_gray(image, width, height, w):
    for x in range(width):
        for y in range(height):
            p_color = image[x, y]
            color = w[0] * p_color[0] + w[1] * p_color[1] + w[2] * p_color[2]
            p_color[0] = p_color[1] = p_color[2] = color
            image[x, y] = p_color
    return image


# 求颜色的平均值函数
def MeanColor(data):
    size = data.shape[0] * data.shape[1]
    r_avg = np.sum(data[:, :, 0]) / size
    g_avg = np.sum(data[:, :, 1]) / size
    b_avg = np.sum(data[:, :, 2]) / size
    return [r_avg, g_avg, b_avg]


def DownSampling(image):
    """
    降采样算法，得到的图片width和 height属性变为原来的一半
    """
    a = 0.6  # 降采样高斯权重
    w = [1.0/4 - a/2.0, 1.0/4, a, 1.0/4, 1.0/4 - a/2.0]
    width = int(image.shape[0] / 2.0 + 0.5)
    height = int(image.shape[1] / 2.0 + 0.5)
    dst_image = np.zeros((width, height, 3), np.uint8)
    x_sample = np.zeros((image.shape[0], height, 3), np.uint8)
    x_padding = np.zeros((image.shape[0], image.shape[1]+4, 3), np.uint8)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if y < 2:
                x_padding[x, y] = image[x, y]
            elif 2 <= y < x_padding.shape[1] - 2:
                x_padding[x, y] = image[x, y-2]
            else:
                x_padding[x, y] = image[x, y-1]
    for x in range(x_sample.shape[0]):
        for y in range(x_sample.shape[1]):
            for m in range(-2, 3, 1):
                x_sample[x, y] = x_sample[x, y] + w[m+2] * x_padding[x, y*2+m+2]

    y_padding = np.zeros((x_sample.shape[0]+4, x_sample.shape[1], 3), np.uint8)
    for i in range(y_padding.shape[0]):
        for j in range(y_padding.shape[1]):
            if i < 2:
                y_padding[i, j] = x_sample[i, j]
            elif i >= 2 and i < y_padding.shape[0] - 2:
                y_padding[i, j] = x_sample[i-2, j]
            else:
                y_padding[i, j] = y_padding[i-1, j]
    for i in range(dst_image.shape[0]):
        for j in range(dst_image.shape[1]):
            for m in range(-2, 3, 1):
                dst_image[i, j] = dst_image[i, j] + w[m+2] * y_padding[i*2+m+2, j]
    return dst_image


def toLab(R, G, B):
    x = 0.412453 * R + 0.357580 * G + 0.180423 * B
    y = 0.212671 * R + 0.715160 * G + 0.072169 * B
    z = 0.019334 * R + 0.119193 * G + 0.950227 * B
    x = x / 0.950456
    z = z / 1.088754
    delta1 = lambda a: 116 * pow(a, 1 / 3) - 16 if a > 0.008856 else 903.3 * a
    delta2 = lambda a: pow(a, 1 / 3) if a > 0.008856 else 7.787 * a + 16 / 116
    l = delta1(y)
    a = 500 * (delta2(x) - delta2(y))
    b = 200 * (delta2(y) - delta2(z))

    return np.array([l / 100, a / 128, b / 128])


# 基于RGB色彩空间的线性投影灰度化算法
class LinearProjection:
    def __init__(self, path):
        self.image = plt.imread(path)
        self.Image_Width, self.Image_Height, self.RGBLength = self.image.shape
        self.im_backup = copy.deepcopy(self.image)
        self.image = np.array(self.image)  # 重新设置该变量的类型，防止其为只读模式而不能修改
        print(self.Image_Width, self.Image_Height)

    # 平均值法
    def average_gray(self):
        image = self.image
        for x in range(self.Image_Width):
            for y in range(self.Image_Height):
                Pcolor = self.image[x, y]
                color = 1.0 / 3 * Pcolor[0] + 1.0 / 3 * Pcolor[1] + 1.0 / 3 * Pcolor[2]
                Pcolor[0] = Pcolor[1] = Pcolor[2] = color
                image[x, y] = Pcolor
        return image

    # 加权平均法1（PPT方法2）
    def weighted_average(self):
        image = self.image
        for x in range(self.Image_Width):
            for y in range(self.Image_Height):
                p_color = self.image[x, y]
                color = 0.2126 * p_color[0] + 0.7152 * p_color[1] + 0.0722 * p_color[2]
                p_color[0] = p_color[1] = p_color[2] = color
                image[x, y] = p_color
        return image

    # 固定权重的线性投影灰度化算法之加权平均法(PPT方法3）
    def luminance(self):
        image = self.image
        for x in range(self.Image_Width):
            for y in range(self.Image_Height):
                p_color = self.image[x, y]
                color = 0.299 * p_color[0] + 0.587 * p_color[1] + 0.114 * p_color[2]
                p_color[0] = p_color[1] = p_color[2] = color
                image[x, y] = p_color
        return image

    # 最大值法
    def max_rgb(self):
        image = self.image
        for x in range(self.Image_Width):
            for y in range(self.Image_Height):
                p_color = self.image[x, y]
                color = max(p_color[0], p_color[1], p_color[1])
                p_color[0] = p_color[1] = p_color[2] = color
                image[x, y] = p_color
        return image

    # 动态权重的线性投影灰度化算法(信息熵最大化)
    def max_entropy(self):
        image = self.image
        # 降低采样率至64*64,实际编程时降低到128*128以下即可
        data = DownSampling(image)
        while data.shape[0] > 128 or data.shape[1] > 128:
            data = DownSampling(data)
        H_m = []
        W_m = []
        for i in range(11):
            for j in range(11 - i):
                w0 = [0.1 * i, 0.1 * j, 1 - 0.1 * i - 0.1 * j]
                image = convert_gray(data, data.shape[0], data.shape[1], w0)  # 得到备选灰度图
                h_m = self.calculate_entropy(image)
                W_m.append(w0)
                H_m.append(h_m)  # 存储66个备选灰度图的信息熵
        max_index = H_m.index(max(H_m))
        w_best = W_m[max_index]
        image_best = convert_gray(self.image, self.image.shape[0], self.image.shape[1], w_best)
        return image_best

    def calculate_entropy(self, image):
        image = np.uint8(image)
        hist_cv = cv2.calcHist([image], [0], None, [256], [0, 256])  # [0,256]的范围是0~255.返回值是每个灰度值出现的次数
        P = hist_cv / (len(image) * len(image[0]))  # 概率
        E = np.sum([-p * np.log2(p + 1e-10) for p in P])  # 防止 P为 0 时直接取对数产生错误
        return E  # 熵


# 实时对比度保留去色算法(RTCP)
def RTCP(image):
    data = DownSampling(image)
    while data.shape[0] > 128 or data.shape[1] > 128:
        data = DownSampling(data)
    width = data.shape[0]
    height = data.shape[1]

    # 随机取64*64组点对
    point_list1_x = np.random.randint(0, width, 64*64)
    point_list1_y = np.random.randint(0, height, 64*64)
    point_list2_x = np.random.randint(0, width, 64*64)
    point_list2_y = np.random.randint(0, height, 64*64)

    def CalculateEnergy(point1_x, point1_y, point2_x, point2_y, w):
        Pcolor1 = data[point1_x, point1_y]
        Pcolor2 = data[point2_x, point2_y]
        lab1 = toLab(Pcolor1[0], Pcolor1[1], Pcolor1[2])
        lab2 = toLab(Pcolor2[0], Pcolor2[1], Pcolor2[2])
        delta = np.linalg.norm(lab1 - lab2, ord=2)
        gray1 = np.dot(w, Pcolor1)
        gray2 = np.dot(w, Pcolor2)
        if all(Pcolor1 > Pcolor2):
            return 50 * pow(gray1 - gray2 - delta, 2)
        elif all(Pcolor2 > Pcolor1):
            return 50 * pow(gray1 - gray2 + delta, 2)
        else:
            return -np.log(0.5 * np.exp(-50 * pow(gray1 - gray2 - delta, 2)) + 0.5 * np.exp(
                -50 * pow(gray1 - gray2 + delta, 2)) + 1e-10)

    best_w = [0, 0, 1]
    least_energy = float("inf")
    for i in range(11):
        for j in range(11 - i):
            w0 = [0.1 * i, 0.1 * j, 1 - 0.1 * i - 0.1 * j]
            energy_w = 0
            for k in range(64*64):
                energy_w += CalculateEnergy(point_list1_x[k], point_list1_y[k],
                                            point_list2_x[k], point_list2_y[k], w0)
            if energy_w < least_energy:
                best_w = w0
                least_energy = energy_w

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            Pcolor = image[x, y]
            color = np.dot(Pcolor[:3], best_w)
            Pcolor[0] = Pcolor[1] = Pcolor[2] = color
            image[x, y] = Pcolor
    return image


# 主成分分析，选取主方向后将其投影在该方向上
def pca(image):
    width, height, depth = image.shape
    size = height * width
    max_origin = np.max(image)
    min_origin = np.min(image)
    # 计算d维矩阵的均值向量（数据集每一通道的均值）
    color_mean = MeanColor(image)

    # 计算矩阵的协方差，注意传入图片的存储形式，可以将其看成是width*height的矩阵，只是矩阵的每个元素都是三元组tuple
    def mul_pixel(color):
        color = np.mat(color - color_mean)
        return np.matmul(color.T, color)

    def mul_row(row):
        return sum(list(map(mul_pixel, row)))

    data = np.array(list(map(mul_row, image)))
    data = sum(data) / size
    # 最大特征值对应的特征向量，即为要找的主方向
    eig_val, eig_vec = np.linalg.eig(data)
    direction = eig_vec[:, np.argmax(eig_val)]
    # 特征向量可能为主方向的反向
    if all(direction < 0):
        direction = -direction
    # 投影
    for x in range(width):
        for y in range(height):
            Pcolor = image[x, y]
            Pcolor = np.array(Pcolor)
            Pcolor[0] = np.dot(direction, Pcolor[0: 3] - color_mean)
    max_new = np.max(image[:, :, 0])
    min_new = np.min(image[:, :, 0])
    k = (max_origin - min_origin) / (max_new - min_new)
    for x in range(width):
        for y in range(height):
            Pcolor = image[x, y]
            Pcolor = np.array(Pcolor)
            color = k * (Pcolor[0] - min_new) + min_origin
            Pcolor[0] = Pcolor[1] = Pcolor[2] = color
            image[x, y] = Pcolor
    return image


# 一种新的基于主成分分析(PCA)的彩色转灰度算法
def pca_new(image):
    size = image.shape[0] * image.shape[1]
    # 从RGB颜色模型变换至YCbCr颜色模型
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            p_color = image[x, y]
            Y = (65.738 * p_color[0] + 129.057 * p_color[1] + 25.06 * p_color[2])/256 + 16
            Cb = (-37.945 * p_color[0] + -74.494 * p_color[1] + 112.43 * p_color[2]) / 256 + 128
            Cr = (112.439 * p_color[0] + -94.154 * p_color[1] + -18.28 * p_color[2]) / 256 + 128
            p_color[0], p_color[1], p_color[2] = Y, Cb, Cr
            image[x, y] = p_color
    # 主成分分析
    ycc_mean = MeanColor(image)
    image = image - ycc_mean

    def mul_pixel(color):
        color = np.mat(color)
        return np.matmul(color.T, color)

    def mul_row(row):
        return sum(list(map(mul_pixel, row)))

    data = np.array(list(map(mul_row, image)))
    data = sum(data) / size
    eig_val, eig_vec = np.linalg.eig(data)
    index = np.argsort(-eig_val)
    # 得到三个特征向量和特征值，并对其归一化
    lambda_1, lambda_2, lambda_3 = eig_val[index[0]], eig_val[index[1]], eig_val[index[2]]
    v_1, v_2, v_3 = eig_vec[index[0]], eig_vec[index[1]], eig_vec[index[2]]
    lambda_norm = np.linalg.norm(eig_val)

    lambda_1, lambda_2, lambda_3 = lambda_1/lambda_norm, lambda_2/lambda_norm, lambda_3/lambda_norm
    v_1, v_2, v_3 = v_1/np.linalg.norm(v_1), v_2/np.linalg.norm(v_2), v_3/np.linalg.norm(v_3)
    v_1, v_2, v_3 = np.mat(v_1), np.mat(v_2), np.mat(v_3)
    # 线性加权得初始灰度图
    image_gray = image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            p_color = image[x, y]
            p_color = np.mat(p_color)
            gray = lambda_1 * np.matmul(v_1, p_color.T) + lambda_2 * np.matmul(v_2, p_color.T)\
                   + lambda_3 * np.matmul(v_3, p_color.T)
            image_gray[x, y][0] = image_gray[x, y][1] = image_gray[x, y][2] = gray[0, 0]
    # 对可能超出0~255的值做缩放
    max_new = np.max(image_gray[:, :, 0])
    min_new = np.min(image_gray[:, :, 0])
    k = 255/(max_new-min_new)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            Pcolor = image_gray[x, y]
            Pcolor = np.array(Pcolor)
            color = k * (Pcolor[0] - min_new) + 0
            Pcolor[0] = Pcolor[1] = Pcolor[2] = color
            image_gray[x, y] = Pcolor
            # 灰度反转
            if abs(int(image[x, y][0])-int(image_gray[x, y][0])) > abs(int(image[x, y][0])+int(image_gray[x, y][0]-255)):
                image_gray[x, y][0] = image_gray[x, y][1] = image_gray[x, y][2] = 255 - image_gray[x, y][0]
    return image_gray


if __name__ == "__main__":
    """
    注：测试代码时应一个模型一个模型地测试（即将其它代码注释掉)，否则上一个模型的颜色会影响下个模型的结果
    """
    # LPImage = LinearProjection("src\\src_picture1.jfif")
    LPImage = LinearProjection("src\\test.jfif")  # 样例图片2
    show_image(LPImage.image)
    # 线性化方法
    # ave_image = LPImage.average_gray()  # 平均值法
    # show_image(ave_image)
    # weight_image = LPImage.weighted_average()  # 加权平均法1
    # show_image(weight_image)
    # lu_image = LPImage.luminance()  # 加权平均法2（luminance算法）
    # show_image(lu_image)
    # max_image = LPImage.max_rgb()  # 最大值法
    # show_image(max_image)
    # me_image = LPImage.max_entropy()  # 动态权重的线性投影灰度化算法(信息熵最大化)
    # show_image(me_image)

    # rtcp_image = RTCP(LPImage.image)  # RTCP算法
    # show_image(rtcp_image)

    pca_image = pca(LPImage.image)  # PCA投影
    show_image(pca_image)

    # pca_new_image = pca_new(LPImage.image)  # 基于PCA的新型灰度化算法
    # show_image(pca_new_image)





