import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
import torch
from pyts.approximation import PiecewiseAggregateApproximation

# 假设你的数据是一个128x60x82的三维数组
# 请替换这里的示例数据为你的实际数据
data = torch.randn(64,16,2560)  # 这里使用随机数据作为示例

# 步骤 3: 计算GAF
def GASF(data,window_size=80):
    # transformer = PiecewiseAggregateApproximation(window_size)
    num_samples,num_features, num_time_steps = data.shape
    num_time_steps=int(num_time_steps/window_size)
    gaf_images = np.zeros((num_samples, num_features, num_time_steps, num_time_steps))

    for i in range(num_samples):
        gasf = GramianAngularField(image_size=num_time_steps, method='difference')
        gaf_images[i] = gasf.fit_transform(data[i])
    gaf_images=torch.tensor(gaf_images).float()
    return gaf_images
gaf_images_gasf = GASF(data)
print(gaf_images_gasf.shape)
# gaf_images_gadf = calculate_gaf(data, method='gadf')

# 步骤 4: 可视化
sample_index = 0  # 选择一个样本进行展示

plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
plt.imshow(gaf_images_gasf[sample_index][sample_index], cmap='viridis', origin='lower')
plt.title(f'GASF Image for Sample {sample_index}')

# plt.subplot(1, 2, 2)
# plt.imshow(gaf_images_gadf[sample_index], cmap='viridis', origin='lower')
# plt.title(f'GADF Image for Sample {sample_index}')

plt.tight_layout()
plt.show()
