import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 随机生成1000个数据
data = pd.Series(np.random.randn(1000),index=np.arange(1000))
 
# 为了方便观看效果, 我们累加这个数据
data = data.cumsum()

# pandas 数据可以直接观看其可视化形式
data.plot()

plt.show()