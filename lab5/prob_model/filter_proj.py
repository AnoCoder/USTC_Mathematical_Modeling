import numpy as np
import matplotlib.pyplot as plt

def filtered_backprojection(sinogram, angles):
    num_angles = len(angles)
    num_detectors = sinogram.shape[0]
    num_pixels = sinogram.shape[1]
    
    reconstructed_image = np.zeros((num_pixels, num_pixels))
    center = num_pixels // 2
    
    for angle_idx in range(num_angles):
        angle = angles[angle_idx]
        projection = sinogram[:, angle_idx]
        
        for detector_idx in range(num_detectors):
            detector_pos = detector_idx - num_detectors // 2
            theta = np.deg2rad(angle)
            x = center + detector_pos * np.sin(theta)
            y = center + detector_pos * np.cos(theta)
            
            for pixel_idx in range(num_pixels):
                pixel_pos = pixel_idx - center
                r = np.sqrt((pixel_pos - x)**2 + (center - y)**2)
                if r > 0:
                    weight = 1 / r
                    reconstructed_image[pixel_idx, pixel_idx] += weight * projection[detector_idx]
    
    return reconstructed_image

# 示例数据
num_pixels = 128
num_detectors = 180
num_angles = 180
angles = np.linspace(0, 180, num_angles, endpoint=False)
sinogram = np.random.rand(num_detectors, num_angles)

# 使用滤波反投影法重建图像
reconstructed_image = filtered_backprojection(sinogram, angles)

# 显示重建结果
plt.imshow(reconstructed_image, cmap='gray')
plt.axis('off')
plt.show()

# (ref)[https://blog.csdn.net/qq_33414271/article/details/78128813]
