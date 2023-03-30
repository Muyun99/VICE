import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


class TwoNomal():
  def __init__(self, mu1, mu2, sigma1, sigma2):
    self.mu1 = mu1
    self.sigma1 = sigma1
    self.mu2 = mu2
    self.sigma2 = sigma2

  def doubledensity(self, x):
    mu1 = self.mu1
    sigma1 = self.sigma1
    mu2 = self.mu2
    sigma2 = self.sigma1
    N1 = np.sqrt(2 * np.pi * np.power(sigma1, 2))
    fac1 = np.power(x - mu1, 2) / np.power(sigma1, 2)
    density1 = np.exp(-fac1 / 2) / N1

    N2 = np.sqrt(2 * np.pi * np.power(sigma2, 2))
    fac2 = np.power(x - mu2, 2) / np.power(sigma2, 2)
    density2 = np.exp(-fac2 / 2) / N2
    # print(density1,density2)
    density = 0.5 * density2 + 0.5 * density1
    return density


N2 = TwoNomal(10, 80, 10, 10)
# 创建等差数列作为X
result = np.random.normal(15, 44, 100) # 均值为0.5,方差为1
print(result)
x = np.arange(min(result), max(result), 0.1)

# print(X)
y = N2.doubledensity(x)

# # 根据均值、标准差,求指定范围的正态分布概率值
# def normfun(x, mu, sigma):
#   pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
#   # pdf = np.random.normal(15, 44, len(x))
#   return pdf
#
#
# # result = np.random.randint(-65, 80, size=100) # 最小值,最大值,数量
# result = np.random.normal(15, 44, 100) # 均值为0.5,方差为1
# print(result)
#
# x = np.arange(min(result), max(result), 0.1)
# # 设定 y 轴，载入刚才的正态分布函数
# print(result.mean(), result.std())
# y = normfun(x, result.mean(), result.std())
# plt.plot(x, y, color='#FF8884') # 这里画出理论的正态分布概率曲线

# 这里画出实际的参数概率与取值关系
n, bins, patches = plt.hist(result, bins=20, rwidth=0.8, density=True, color='#9AC9DB') # bins个柱状图,宽度是rwidth(0~1),=1没有缝隙


x_plot = bins[0:20]+((bins[1]-bins[0])/2.0)

y_plot = n
y_smoothed = gaussian_filter1d(y_plot, sigma=3)
# print(n)
plt.plot(x_plot, y_smoothed, color='#FF8884')#利用返回值来绘制区间中点连线

# plt.title('distribution')
# plt.xlabel('temperature')
# plt.ylabel('probability')
# 输出
plt.axis('off')
plt.savefig('distribution')
# plt.show()
