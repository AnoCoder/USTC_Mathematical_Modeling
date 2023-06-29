import numpy as np
import matplotlib.pyplot as plt


def zip_image_by_svd(origin_image, rate=0.8):
    # 显示原图像
    plt.figure(figsize=(12, 12))
    plt.title("Origin Image")
    plt.imshow(origin_image)
    plt.axis('off')  # 去掉坐标轴
    plt.show()
    result = np.zeros(origin_image.shape)

    # 对原图像进行SVD分解
    u_shape = 0
    s_shape = 0
    vT_shape = 0

    for chan in range(3):
        # 对该层进行SVD分解
        U, sigma, V = np.linalg.svd(origin_image[:, :, chan])
        n_sigmas = 0
        temp = 0

        # 计算达到保留率需要的奇异值数量
        while (temp / np.sum(sigma)) < rate:
            temp += sigma[n_sigmas]
            n_sigmas += 1

        # 构建奇异值矩阵
        S = np.zeros((n_sigmas, n_sigmas))

        for i in range(n_sigmas):
            S[i, i] = sigma[i]

        # 构建结果
        result[:, :, chan] = (U[:, 0:n_sigmas].dot(S)).dot(V[0:n_sigmas, :])
        u_shape = U[:, 0:n_sigmas].shape
        s_shape = S.shape
        vT_shape = V[0:n_sigmas, :].shape

    # 归一化到[0, 1]
    for i in range(3):
        MAX = np.max(result[:, :, i])
        MIN = np.min(result[:, :, i])
        result[:, :, i] = (result[:, :, i] - MIN) / (MAX - MIN)

    # 调整到[0, 255]
    result = np.round(result * 255).astype('int')

    # 计算压缩比
    zip_ratio = origin_image.size / (3 * (u_shape[0] * u_shape[1] + s_shape[0] * s_shape[1] + vT_shape[0] * vT_shape[1]))
    print("保留率：        ", rate)
    print("所用奇异值数量为：", n_sigmas)
    print("原图大小：       ", origin_image.shape)
    print("压缩后用到的矩阵大小：3 x ({} + {} + {})".format(u_shape, s_shape, vT_shape))
    print("压缩比为：       ", zip_ratio)
    # 显示压缩结果
    plt.figure(figsize=(12, 12))
    plt.imshow(result)
    plt.title("Result Image")
    plt.axis('off')  # 去掉坐标轴
    plt.show()


if __name__ == "__main__":
    # image_path = '../assets/image/image_bmp/lena.bmp'
    image_path = '../assets/image/image_bmp/sunrise.bmp'
    # image_path = '../assets/image/image_jpg/peppers_color.jpg'
    origin_image = plt.imread(image_path)
    zip_image_by_svd(origin_image, rate=0.9)
