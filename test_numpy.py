#-*- coding:utf-8 -*-

import numpy as np
from numpy.core.fromnumeric import reshape
# from numpy.core.records import array
import matplotlib.pyplot as mp

# #特点：1.元数据和实际数据分开  2.存储同质(元素类型一样)数据
print("创建数组")
ary = np.array([1,2,3,4,5,6]) 
# a=np.arange(0,10,2)                #[0 2 4 6 8] 
# b=np.zeros(10,dtype="int32")       #[0 0 0 0 0 0 0 0 0 0]
# c=np.ones(3,dtype="bool")          #[ True  True  True ]
# d=np.ones((2,3),dtype="float32")   #[[1. 1. 1.] [1. 1. 1.]]
# e = np.ones(5) / 5                 #[0.2 0.2 0.2 0.2 0.2]
# f = np.zeros_like(d)               #维度和d一样，[[0. 0. 0.] [0. 0. 0.]] 
# # print(a,b,c,d,e,f) 
print(ary,type(ary))  #[1 2 3 4 5 6] <class 'numpy.ndarray'>
# print(ary.shape)      #维度，(6,) 一维数组，6个元素
# ary.shape=(2,3)     #维度改为2行3列
# print(ary,ary.shape)  # [[1 2 3] [4 5 6]]   (2, 3)
# ary.shape=(6,)
# print('数组的运算,针对数组每个元素（for）')
# print(ary + 3)    #[4 5 6 7 8 9]
# print(ary - 3)    #[-2 -1  0  1  2  3]
# print(ary * 3)    #[ 3  6  9 12 15 18]
# print(ary / 3)    #[0.33333333 0.66666667 1.         1.33333333 1.66666667 2.        ]
# print(ary // 3)   #[0 0 1 1 1 2]

# print(ary > 3)    #[False False False  True  True  True]
# print(ary >= ary)
# print(ary + ary)  #[ 2  4  6  8 10 12]
# print(ary - ary)  #[0 0 0 0 0 0]
# print(ary * ary)  #[ 1  4  9 16 25 36]
# print(ary / ary)  #[1. 1. 1. 1. 1. 1.]
# print(ary // ary)  #[1 1 1 1 1 1]

# print('属性')
# k = np.arange(1,9)
# print(dir(k))
# print('维度基础操作')
# print(k,k.shape)    #返回元组的乘积等于size属性的值
# print('数据类型基础操作')
# print(k.dtype)
# k.dtype = "float32"  #错误的
# print(k,k.dtype)
# l = k.astype('float32')   #正确的
# print(l,l.dtype)
# print('size属性')
# m = np.arange(1,9)
# m.shape = (2,4)
# print(m, 'size:',m.size,'length',len(m))   #size: 8 length 2

# print('索引 下表 index操作')
# c=np.arange(1,25)
# c.shape = (3,2,4)   # 3页2行4列，倒着读，最小的列表有四个元素，两个小列表合成一个中列表，三个中列表合成一个大列表
#                     # z=3 y=2 x=4
# print(c)
# '''
# [[[ 1  2  3  4]    #开头有几个中括号代表几维
#   [ 5  6  7  8]]

#  [[ 9 10 11 12]
#   [13 14 15 16]]

#  [[17 18 19 20]
#   [21 22 23 24]]]'''
# print(c[0][1][0]) #5,大列表的0元素列表，中列表的1元素，小列表的0元素
# print(c[0,1,0])



# print('复合类型, 封装成对象的思想，只是不用常规的语法')
# data = [('zs',[90,80,85],15),
# ('ls',[92,81,83],16),
# ('ww',[95,85,95],15)]
# print('第一种设置的方式')
# a = np.array(data,dtype='U2,3int32,int32')  #2个字符(2个Unicode，3个int32，int32)
# print(a)
# print(a[1][1])
# print('第二种设置的方式，为字段起别名')
# b=np.array(data,
# dtype=[('name','str',2),
# ('scores','int32',3),
# ('age','int32',1)])
# print(b)
# print(b[2]["scores"])
# print('第三种设置dtype的方式')
# c=np.array(data,dtype={'names':['name','scores','age'],'formats':['U2','3int32','int32']})
# print(c)
# print(c[1]['name'])
# print('第四种设置dtype的方式,0,16,28表示内存偏移量')  
# d = np.array(data, dtype={'names': ('U3', 0),
#                     'scores': ('3int32', 16),
#                     'ages': ('int32', 28)})
# print(d[0]['names'], d[0]['scores'], d.itemsize)
# print('第五种设置dtype的方式，大端字节序和小端字节序')
# e = np.array([0x1234, 0x5667],
#              dtype=('u2', {'lowc': ('u1', 0),
#                             'hignc': ('u1', 1)}))
# print('%x' % e[0])
# print('%x %x' % (e['lowc'][0], e['hignc'][0]))

# print('日期数据类型')
# dates = ['2011','2012-01','2013-01-01','2014-01-01 00:00:01']
# dates=np.array(dates)
# dates=dates.astype('M8[s]')  #精确到秒
# dates=dates.astype('M8[D]')  #精确到日期
# print(dates,dates.dtype)
# print(dates[2] - dates[1])    #相差多少天
# dates = dates.astype('int32')
# print("dates",dates)          # [14975 15340 15706 16071] 表示从1970.1.1到现在的天数，和上面精确度有关

# print('数据变维，直接修改shape也可以')
# print('1.rershape和ravel有返回值，元数据独立，数据共享/原数组不变)')
# a=np.arange(1,10)
# print(a,a.shape)
# b=a.reshape(3,3)      #变成3行3列数组
# print('a',a)
# print('b',b)
# print('c',b.ravel())  #撑平，即变为1为数组
# a[0]=999
# print('a',a)
# print('b1',b)
# print('c1',b.ravel())

# print('有返回值，复制一份数据')
# # d=a.copy()
# d=a.flatten()
# print(d)
# a[0]=88
# print(a,d)

# print('没有返回值，修改元组本身')
# a.shape=(3,3)
# print(a)
# a.resize((9,))
# print(a)


# print('数组切片')
# print('一维数组和列表一样用法')
# import numpy as np
# a = np.arange(1, 10)
# print(a)  # 1 2 3 4 5 6 7 8 9
# print(a[:3])  # 1 2 3
# print(a[3:6])   # 4 5 6
# print(a[6:])  # 7 8 9
# print(a[::-1])  # 9 8 7 6 5 4 3 2 1
# print(a[:-4:-1])  # 9 8 7
# print(a[-4:-7:-1])  # 6 5 4
# print(a[-7::-1])  # 3 2 1
# print(a[::])  # 1 2 3 4 5 6 7 8 9
# print(a[:])  # 1 2 3 4 5 6 7 8 9
# print(a[::3])  # 1 4 7
# print(a[1::3])  # 2 5 8
# print(a[2::3])  # 3 6 9
# print('维度用逗号隔开（二维一个逗号分开，三维用两个逗号分开,一次类推）')
# a.shape=(3,3)
# print(a)
# print(a[:2,])    #获取前两行
# print(a[:2,:2])  #前两行的前两列
# '''
# [[1 2]
#  [4 5]]
# '''
# print(a[::2,:])  #1，3行
# '''
# [[1 2 3]
#  [7 8 9]]
# '''
# print(a[::2,::-1])  #1，3行
# '''
# [[3 2 1]
#  [9 8 7]]
#  '''
# print(a[:,0])   #第一列，[1 4 7]

# print('基于bool数组的掩码')
# a=np.arange(1,10)
# mask = [True, False,True, False,True, False,True, True, True]
# print(a)
# print(a[mask])    # 返回True的值 [1 3 5 7 8 9]
# print(a%3==0)      #里面是一个比较运算，返回True，False组合成的数组[False False  True False False  True False False  True]
# print(a[a%3==0])   #返回[3 6 9]

# b=np.arange(1,100)
# # mask = (b % 3 == 0) and (b%7==0)   #这是错的，and是两个值的运算
# mask = (b % 3 == 0) & (b%7==0)       #按位与
# print(mask)                          #返回100个True，False组合成的数组
# print(b[mask])                       #返回相同索引对应的值[21 42 63 84]

# print('基于索引的掩码')
# names = np.array(['apple','mate30','Mi','oppo','vivo'])
# rank=[1,0,3,4,2]     #排名
# print(names[rank])   #按排名输出['mate30' 'apple' 'oppo' 'vivo' 'Mi']

# print('多数组的组合和拆分')
# a = np.arange(1,7).reshape(2,3)
# b = np.arange(7,13).reshape(2,3)
# print('初始输出')
# print(a,b)
# '''
# [[1 2 3][4 5 6]] 
# [[ 7  8  9][10 11 12]]
# '''
# print('水平方向操作（2行6列）')
# c=np.hstack((a,b))
# print(c)
# '''
# [[ 1  2  3  7  8  9]
#  [ 4  5  6 10 11 12]]
# '''
# a,b=np.hsplit(c,2)    #拆分
# print(a,b)
# print('垂直方向操作（4行3列）')
# c=np.vstack((a,b))
# print(c)
# '''
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]
# '''
# a,b=np.vsplit(c,2)   #垂直拆分
# print(a,b)
# print('深度方向操作,(2页3行2列)')
# c=np.dstack((a,b))
# print(c.shape)
# '''
# [[[ 1  7]
#   [ 2  8]
#   [ 3  9]]
#  [[ 4 10]
#   [ 5 11]
#   [ 6 12]]]
# '''
# a,b=np.dsplit(c,2)
# print(a,b)
# print('长度不等的数组组合')
# a=np.array([1,2,3,4,5])
# b=np.array([1,2,3,4])
# # (0,1) 头部补0个元素，尾部补1个元素，mode是常量，值为-1
# b=np.pad(b,pad_width=(0,1),mode='constant',constant_values=-1)
# print(b)
# c=np.vstack((a,b))
# print(c)
# print('一维数组组合')
# print(np.row_stack((a,b)))    #按行合并
# print(np.column_stack((a,b))) #按列合并

# a=np.array([[1+1j,2+4j,3+7j],
#             [4+2j,5+5j,6+8j],
#             [7+3j,8+6j,9+9j]])
# print(a.shape)
# print(a.dtype)
# print(a.itemsize)     # 16字节
# print(a.size)         # 9个元素
# print(a.nbytes)       
# print(a.real)
# print(a.imag)
# print(a.T)
# print([i for i in a.flat])
# for i in a.flat:
#     print(i)

# print('numpy读取文件数据')
# import datetime as dt
# def dmy2ymd(dmy):
#   dmy = str(dmy,encoding='utf-8')
#   time = dt.datatime.strptime(dmy,'%d-%m-%Y').datetime()
#   t = time.strftime('%Y-%m-%d')
#   return t
# dates,opening_prices,highest_prices,\
#   lowest_prices,closing_prices=np.loadtxt(     #npmpy加载文件，将返回数据存到对应变量中
#     'path/*.csv',delimiter=',',                #相对路径，分隔符号是,
#     usecols=(1,3,4,5,6),                       #获取第1,3,4,5,6列
#     dtype='M8[D],f8,f8,f8,f8',                 #每一列的类型
#     unpack=True,                               #将读取到的数据分一个数组存储(False)还是分为5个数组存(True)
#     convertes={1:dmy2ymd}                      #将第一列的数据传给函数处理再存储
#   )



'''
np.max(): 最大值
np.min()：最小值
np.ptp()：极差：最大值－最小值
np.argmax(): 最大值的下标
mp.argmin()：最小值的下标
np.maximum(): 将两个同为数组中对应的最大元素构成一个新的数组
np.minmun(): 将两个同为数组中对应的最小元素构成一个新的数组
'''

'''
np.msort(data)  #对数组进行排序
print('中位数')
median=np.median(prices)
'''

'''
print('算数平均值，表示对真值(真正价值)的无偏估计')
1. np.mean(datas)
2. datas.mean()

print('加权平均值,意义比算数平均值更接近真值')
# 成交量体现了市场对当前交易价格的认可度(有买有卖才能交易)
np.average(arr_prices,weights=arr_weight)

mean = np.mean(data)                   # 算数平均值
devs = datas- data                     # 离差：真实值减去平均值(有正负的数组)
deqs = devs ** 2                       # 离差方：让离差每个元素求平方(只有正的数组)
pvar = np.sum(dsqs) / dsqs.size        # 总体方差：对离差方数组求和除以数量(sum/n是一个指标值)
pstd = np.sqrt(pvar)                   # 总体标准差：对总体方差求开方(是一个指标值，表示一个线的震荡幅度)
svar = np.sum(dsqs) / (dsqs.size - 1)  # 样本方差：对离差方数组求和除以数量-1(sum/n-1是一个指标值)
sstd = np.sqrt(svar)                   # 样本标准差：对样本方差求开方(是一个指标值)

cov_ab = np.mean(dev_a * dev_b)         # 协方差：离差相乘，表示两组数据的相关性(指标值：正相关与负相关)                                    

相关性系数：协方差除以两组统计样本标准差的乘积是一个[-1,1]之间的数，该结果称为统计样本的相关系数
coef =cov_ab / (np.std(data1)*np.std(data2))
若值接近0，表示越不相关
接近1，表示正相关
接近0，表示负相关
相关矩阵：np.corrcoef(data1,data2)
'''


'''
# 按轴进行汇总
def apply(data):
  return data.mean()
r = np.apply_along_axis(
  apply, #汇总函数
  0,     #0表示垂直方向，1表示水平方向
  ary    #需要汇总的数据集
)
'''

'''
# 卷积类型
从运算过程来看：一个值和相邻元素都有关系
现实意义：一个结果和周围元素都有关系
分类：
  有效卷积valid: 所有卷积核元素都参与运算，结果比原始数据少
  同维卷积same: 卷积核中心元素参与运算，结果和原始数据一样多
  完全卷积full: 卷积核元素有一个参与运算，结果比原始数据多
# 卷积函数实现5日均线
sma5 = np.convolve(datas,np.ones(5)/5,'valid')
# 加权卷积
weights = np.exp(np.linspace(-1,0,5))
weights /= weights.sum()   #防止卷积后数太大
ema5 = np.convolve(datas,weights[::-1],'valid')
'''

'''
布林带：有一条压力线和一条支撑线
'''

'''
# 点乘：两数组对应位置相乘再相加
# B.dot(A)
'''

''' 
import matplotlib.pyplot as mp
#线性拟合
x=np.array([[1,2,3],[2,1,3],[2,1,4],[2,1,4]])    # 一行有多少个元素表示有多少个一次变量，这边表示有三个变量x，y，z
y=np.array([6,6,7,8])                            # 函数值数量要和x的列数保持一致
P=np.linalg.lstsq(x,y)[0]
print(P)   # a=P[0],b=P[1],c=P[2]                # 拟合出来的线性函数系数a，b，c

# 多项式拟合
# 获得多项式拟合系数
P = np.polyfit(x,y,4)   # 表示将输入的x和y数组拟合成最高次数为4的多项式
# 多项式相关的API
# 例如 y = 4*x**3+3*x**2-1000*x+1
P=[4,3,-1000,1]
X=np.linspace(-20,20,1000)
Y=np.polyval(P,X)    #求出所有点的函数值
# 根据拟合系数求出多项式函数导函数的系数
Q = np.polyder(P)        # 输入多项式系数数组P
print(Q)     # 返回导函数的系数
# 已知多项式系数Q，求多项式函数的根 （y=0时的x坐标/与x轴交点的横坐标）
xs=np.roots(P)
print(xs)
# 根据你和系数与自变量求出拟合值
ys=np.polyval(P,xs)   # 输入多项式系数数组P和自变量数组X
print(ys)   # 返回函数值
# 零个多项式函数的差函数的系数
# 场景：通过查函数的根求取两个曲线的交点
P1=[3,3,-10,1]
Q1=np.polysub(P,P1)   #输入两个多项式的系数
print(Q1)             #返回差函数的系数[1 0 0 0]
# 画图
mp.scatter(xs,ys,color='red',s=60)
mp.plot(X,Y)
y0=np.zeros(1000)
mp.plot(X,y0)
mp.show()
'''

'''
# 符号数组
# 负数映射为-1，0映射为0，正数映射为1
data = np.array([-8,7,0,4,-9])
ary = np.sign(data)
print(ary)    #[-1  1  0  1 -1]
# 判断
ary = np.array([70,80,60,30,40,90])
d = np.piecewise(
  ary,
  [ary < 60, ary==60,ary>60],
  [-1,0,1]
)
print(d)   #[ 1  1  0 -1 -1  1]
'''
'''
# 矢量化: 用数组代替标量来操作数组里的每个元素
# numpy提供了vectorize函数，可以把处理标量的函数矢量化，返回的函数可以直接处理ndarray数组
import math as m
def foo(x,y):
  #如果是np.sqrt(x**2+y**2) 则是可以直接传入数组的
  return m.sqrt(x**2+y**2)
x,y = 3,4
print(foo(x,y))
x,y = np.array([3,4,5]),np.array([4])
x1,y1 = np.array([3,4,5]),np.array([4,5,6])
# 矢量化foo函数 返回矢量化函数
foo_vec = np.vectorize(foo)
print(foo_vec(x,y))
print(foo_vec(x1,y1))
'''

'''
矩阵式numpy.matrix类 类型的对象，该类继承自bumpy.ndarray 
--->  任何针对多维数组的操作,对矩阵同样有效，但是作为子类矩阵又结合其自身的特点，做了必要的扩充，比如：乘法计算，求逆等。
'''

'''
# 矩阵对象的创建
ary = np.arange(1,10).reshape(3,3)
print(ary,ary.shape,type(ary))
m1 = np.matrix(
  ary,     #任何可被解释为矩阵的二维容器  
  copy=True  #缺省值维True，表示数据不共享
)
print(m1,m1.shape,type(m1))
m2 = np.mat('1 2 3;4 5 6')   #数据共享
print(m2)

print('矩阵乘法')
print(ary * ary)    #数组乘法
print(m1 * m1)      #矩阵乘法
print(np.dot(m1,m1))

print('矩阵的逆矩阵')
# 若两个矩阵A，B满足： AB = BA = E ( E为单位矩阵AE = EA =A ) ，则称A为B(B为A)的逆矩阵
# 场景：就像6和1/6的关系一样，可用于求解（3x=6 ==> x=1/3*6）
e = np.mat('1 2 6;3 5 7;4 8 9')
print(e)
print(e.I)
print(e * e.I)
# 把逆矩阵的定义推广到非方阵，称为广义逆矩阵
e = e[:2,:]
print(e)
print(e.I)
print(e * e.I)
# 解方程
prices=np.mat('3 3.2;3.5 3.6')
totals=np.mat('118.4;135.2')
#使用np提供的API解方程
print(np.linalg.lstsq(prices,totals)[0])    #拟合求解
print(np.linalg.solve(prices,totals))       #精确解方程
x = prices.I * totals
print(x)

print('输出斐波那契数列')
n=32
F = np.mat('1 1;1 0')
for i in range(1,n):
  print((F**i)[0,0])
print(F**0)
'''

'''
print('通用函数')
a=np.arange(1,10)
print(a)
print('裁剪,保证数组取值在一定范围内')
print(np.clip(a,3,7))
print(a.clip(min=3,max=7))
print('压缩，只保留a>5的元素')
print(a.compress(a>5))
print(a[a>5])
# 3<a<7
mask = np.all([a>3,a<7],axis=0)    #根据条件返回掩码(True,False)
print(mask)
print(a.compress(mask))
print(a[mask])
print('加法与乘法通用函数')
a=np.arange(1,10)
print(a)
print(np.add(a,a))     #两数组对应元素相加，返回一个数组
print(np.add.reduce(a)) #一个数组累加，返回一个值
print(np.add.accumulate(a))   # 累加过程，返回一个数组，索引表示前几个元素和
print(a.prod())       #一个数组累乘，返回一个值
print(a.cumprod())    # 累乘过程，返回一个数组，索引表示前几个元素积
print(np.add.outer([10,20,30],a))   #外加，即数组a分别加上[10,20,30]，返回三个数组
print(np.outer([10,20,30],a))       #外积

print('除法')
a = np.array([20,20,-20,-20])
b=np.array([3,-3,6,-6])
print(np.divide(a,b))   # a真除b，相当于a/b ，面向对象的喜欢用
print(np.floor(a/b))    # a地板除b ,向下取整
print(np.ceil(a/b))     # a天花板除b，向上取整
print(np.trunc(a/b))    # a截断除b，去除小数点后的数字
print(np.round(a/b))    # 四舍五入

print('按位异或，对应位相同(都为0或都为1)为0，不同为1')
# https://www.bilibili.com/video/BV1Fv41157dG?p=71&spm_id_from=pageDriver
# 场景：判断正负（正数第一位为0，负数第一位为1）
a = np.array([-9,-5,2,1,-9])
b = np.array([-2,6,-2,3,-4])
print(a^b)    #值不重要，重要的是正(同为负数或同为正数)负(一正一负)
print(np.bitwise_xor(a,b))
c=a^b
print(np.where(c<0))
print(np.where(c<0)[0])

print('按位与，都为1才为1')
# 场景：判断是否为2的n次方， n & (n-1) == 0
# 场景：判断奇偶，n & 1
print('位或，相同位只要有一位1，则最终为1')
print('右移1位即除以2的1次方；并且取整')
print('左移一位，相当于乘以2的一次方')
'''


'''
傅里叶定理：任何一条周期曲线，无论多么跳跃或不规则，都能表示成一组光滑正弦曲线叠加之和
'''
'''
# 合成方波(城墙一样的周期波)
import matplotlib.pyplot as mp
x=np.linspace(0,np.pi*4,1000)
y1=4*np.pi*np.sin(x)
y2=4/3*np.pi*np.sin(3*x)
y3=4/5*np.pi*np.sin(5*x)
# y=y1+y2+y3
#叠加1000条曲线
n=1000
y=np.zeros(n)
for i in range(1,n+1):
  y += 4/(2*i-1)*np.pi*np.sin((2*i-1)*x)

mp.grid(linestyle=':')
mp.plot(x,y1,label='y1')
mp.plot(x,y2,label='y2')
mp.plot(x,y3,label='y3')
mp.plot(x,y,label='y')
mp.legend()
mp.show()
'''

'''
#特征值和特征向量：对于n阶方阵A(n组)，如果存在数a和非零n维列向量x，使得Ax=ax，则称a是矩阵A的一个特征值，x是矩阵A属于特征值a的特征向量
A=np.mat('3 4 6;2 3 -1;5 7 11')
print(A)
eigvals,eigvecs = np.linalg.eig(A)
print(eigvals)    #特征值数组  
print(eigvecs)    #特征向量数组
#求原方阵
A2 = eigvecs*np.diag(eigvals)*eigvecs.I
print(A2)
'''

'''
奇异值：矩阵M，可以分解为3个矩阵U，S，V，使得U*S*V等于M。
U与V都是正交矩阵（乘以自身的转置矩阵结果为单位矩阵）
那么S矩阵主对角线上的元素称为矩阵M的奇异值，其它元素均为0
'''
'''
print('奇异值')
M=np.mat('4 11 14; 8 7 -2')
print(M)
#奇异值分解，返回的sv保存所有奇异值
U,sv,V = np.linalg.svd(M,full_matrices=False)
#U,V是正交矩阵
print(U*U.T)
print(V*V.T)
print(U.shape,sv.shape,V.shape)
# 推到原矩阵
M2 = U * np.diag(sv) * V
print(M2)
# 抹除部分奇异值
sv[1:]=0
M3 = U * np.diag(sv) * V
print(M3)
'''

'''
傅里叶变换的目的是将
时域(时间域，即横坐标为时间，纵坐标为位移)上的信号转换为
频域(频率域，即横坐标为频率，纵坐标为能量(振幅))上的信号
随着域的不通，对同一事物的了解角度也就随之改变
既方便处理，又减少存储量
'''
'''
傅里叶变换相关函数
导入快速傅里叶变换所需模块
import numpy.fft as nf
通过采样数与采样周期求得傅里叶变换分解所得曲线的频率序列
freqs = np.fft.fftfreq(采样数量，采样周期)
通过原函数值的序列经过快速傅里叶变换得到一个复数数组，
复数的模代表的是振幅，复数的辐角代表初相位
np.fft.fft(原函数值序列) --->  目标函数值序列(复数)
通过一个复数数组经过你想傅里叶变换得到合成的函数值数组
'''

'''
基于傅里叶变换的频域滤波：
含噪信号是高能信号与低能噪音声叠加的信号，可以通过傅里叶变换的频域滤波实现降噪
通过fft使含噪信号转换成含噪频谱，去除低能噪声，留下高能频谱后再通过ifft留下高能信号
'''

'''
# 案例
import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import matplotlib.pyplot as mp

#得到采样率(每秒采样点的个数)和采样位移(采样点的数值)
sample_rate, noised_sigs = wf.read('../data/noised.wav')
print(sample_rate, noised_sigs.shape)    # 44100 (225000,)
noised_sigs = noised_sigs / 2 ** 15   # 纵坐标值太大，按比例缩小成1
times = np.arange(noised_sigs.size) / sample_rate   #得到时间(秒)
#画出时域图(时间/位移)
mp.figure('Filter', facecolor='lightgray')
mp.subplot(221)
mp.title('Time Domain', fontsize=16)
mp.ylabel('Noised Signal', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(times[:178], noised_sigs[:178],c='orangered', label='Noised')
mp.legend()
# mp.tight_layout()


# 基于傅里叶变换，获取音频频域信息
freqs = nf.fftfreq(times.size, times[1]-times[0])
complex_ary = nf.fft(noised_sigs)   # 得到复数数组
noised_pows = np.abs(complex_ary)   # 求复数的模，即能量
# 绘制音频频域的：频率/能量图像
mp.subplot(222)
mp.title('Frequency Domain', fontsize=16)
mp.ylabel('Power', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.semilogy(freqs[freqs >= 0],noised_pows[freqs >= 0], c='limegreen',label='Noised')  #纵轴是2**n
mp.legend()

#将低能噪声去除后绘制音频频域的：频率/能量图像。
fund_freq = freqs[noised_pows.argmax()]   #找到能量最高的正弦函数的频率
noised_indices = np.where(freqs != fund_freq)  #不是高能的下标，即噪声下标
filter_ffts = freqs.copy()
filter_ffts[noised_indices] = 0        # 将噪声去除，即将低能下标置0
filter_pows = np.abs(filter_ffts)
mp.subplot(224)
mp.xlabel('Frequency', fontsize=12)
mp.ylabel('Power', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(freqs[freqs >= 0], filter_pows[freqs >= 0],c='dodgerblue', label='Filter')
mp.legend() 

# 基于滤波后的复数数组，做逆向傅里叶变换，生成新的音频信号
filter_sigs = nf.ifft(filter_ffts).real
#绘制音频时域的：时间/位移图像。
mp.subplot(223)
mp.xlabel('Time', fontsize=12)
mp.ylabel('Signal', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(times[:178], filter_sigs[:178],c='hotpink', label='Filter')
mp.legend()

#重新生成音频文件。sample_rate / 2 则生成原音频2倍的时间音频
wf.write('../../data/filter.wav',sample_rate,(filter_sigs * 2 ** 15).astype(np.int16))

mp.show()
'''

'''
# 随机数
# 二项式：重复n次独立事件的伯努利实验，在每次实验中只有两种结果，且两种结果相互独立，相互对立
# 产生size个随机数，每个随机数来自nci尝试中的成功次数，其中每次尝试成功的概率为P
# np.random.binomial(n,p,size)   返回size个元素的数组
# 某人打客服电话，客服接通率是0.6，一共打了三次都没人接的概率
print(sum(np.random.binomial(3,0.6,20000) == 0) / 20000)   #np.random.binomial(3,0.6,20000) 返回20000个元素的数组，==0数组掩码，sum对数组求和，true表示1，false表示0

# 超几何分布：产生size个随机数，每个随机数t为在总样本中随机抽取nsample个样本后好样本的个数，总样本由ngood个好样本和nbad个坏样本组成
# 场景，在一个黑盒子里摸出10个球，问分别有几个黑球和红球
# np.random.hypergeometric(ngood,nbad,nsample,size)
# 7个好苹果，3个坏苹果，随机抽3个，求概率
a = np.random.hypergeometric(7,3,3,100000)
print('P(0)',(a==0).sum()/100000)

正态分布(normal)
# 产生size个随机数，服从标准正态(期望=0, 标准差=1)分布。
np.random.normal(size)
# 产生size个随机数，服从正态分布(期望=1, 标准差=10)。
np.random.normal(loc=1, scale=10, size)
'''

'''
# 排序：np.msort(closing_prices)
# 联合间接排序
# 联合间接排序支持为待排序序列排序，若待排序序列值相同，则利用参考序列作为参考继续排序，最终返回排序过后的有序索引序列
# indices = numpy.lexsort((次排序序列,主排序序列))
names = np.array(
  ['Apple','Huawei','Mi','Oppo','Vivo']
)
prices = np.array([8888,5888,2999,3999,3999])
volumes = np.array([60,110,40,50,70])
# 先按价格升序排序，再按销量降序排序
indices= np.lexsort((-volumes,prices))
print(names[indices])

# 复数数组排序
# 按照实部的升序排序，对于实部相同的元素，参考虚部的升序，直接返回排序后的结果数组
# numpy.sort_complex(复数数组)

# 插入排序
# 向有序数组中插入元素，使数组依然有序，numpy提供了searchsorted方法查询(有序数组)并返回可插入位置数组。
# 向 [1,2,4,6] 插入[3,7] 返回索引
a = np.array([1,2,4,6])
b = np.array([3,7])
indices = np.searchsorted(a,b)
# 调用numpy提供了insert方法将待插序列中的元素，按照位置序列中的位置，插入到被插序列中，返回插入后的结果。
c = np.insert(a, indices, b)
print(c)
'''

'''
# 插值：插入值
# 插值算法可以通过一组散点得到一个符合一定规律插值器函数，若给插值器函数散点x坐标序列，该函数将会返回相应的y坐标序列
# 即离散数据连续化
# func = scipy.interp1d(
#   离散水平坐标，
#   离散垂直坐标，
#   kind=插值算法(缺省为线性插值)
# )
import scipy.interpolate as si
# 原始数据 11组数据
min_x = -50
max_x = 50
dis_x = np.linspace(min_x, max_x, 11)
dis_y = np.sinc(dis_x)
# 通过一系列的散点设计出符合一定规律插值器函数，使用线性插值（kind缺省值）
linear = si.interp1d(dis_x, dis_y)
lin_x = np.linspace(min_x, max_x, 200)
lin_y = linear(lin_x)
mp.plot(lin_x,lin_y,color='green',label='linear interpolation')
# 三次样条插值 （CUbic Spline Interpolation） 获得一条光滑曲线
cubic = si.interp1d(dis_x, dis_y, kind='cubic')
cub_x = np.linspace(min_x, max_x, 200)
cub_y = cubic(cub_x)
mp.plot(cub_x,cub_y,color='orangered',label='cubic interpolation')

mp.legend()
mp.show()
'''


'''
微分：
积分
  定积分：根据求曲边梯形的面积得出来的，即得到一个数
  不定积分：求出一个原函数F（x）,使得F'(x)=f(x)，而F(x)+C（C为任意常数）就是不定积分∫f'(x)dx的所有原函数
import scipy.integrate as si
# 利用quad求积分 给出函数f，积分下限与积分上限[a, b]   返回(积分值，最大误差)
area = si.quad(f, a, b)[0]
print(area)
'''

'''
# 金融相关
import numpy as np
# 终值 = np.fv(利率, 期数, 每期支付, 现值)
# 将1000元以1%的年利率存入银行5年，每年加存100元，
# 到期后本息合计多少钱？
fv = np.fv(0.01, 5, -100, -1000)
print(round(fv, 2))
# 现值 = np.pv(利率, 期数, 每期支付, 终值)
# 将多少钱以1%的年利率存入银行5年，每年加存100元，
# 到期后本息合计fv元？
pv = np.pv(0.01, 5, -100, fv)
print(pv)
# 净现值 = np.npv(利率, 现金流)
# 将1000元以1%的年利率存入银行5年，每年加存100元，
# 相当于一次性存入多少钱？
npv = np.npv(0.01, [
    -1000, -100, -100, -100, -100, -100])
print(round(npv, 2))
fv = np.fv(0.01, 5, 0, npv)
print(round(fv, 2))
# 内部收益率 = np.irr(现金流)
# 将1000元存入银行5年，以后逐年提现100元、200元、
# 300元、400元、500元，银行利率达到多少，可在最后
# 一次提现后偿清全部本息，即净现值为0元？
irr = np.irr([-1000, 100, 200, 300, 400, 500])
print(round(irr, 2))
npv = np.npv(irr, [-1000, 100, 200, 300, 400, 500])
print(npv)
# 每期支付 = np.pmt(利率, 期数, 现值)
# 以1%的年利率从银行贷款1000元，分5年还清，
# 平均每年还多少钱？
pmt = np.pmt(0.01, 5, 1000)
print(round(pmt, 2))
# 期数 = np.nper(利率, 每期支付, 现值)
# 以1%的年利率从银行贷款1000元，平均每年还pmt元，
# 多少年还清？
nper = np.nper(0.01, pmt, 1000)
print(int(nper))
# 利率 = np.rate(期数, 每期支付, 现值, 终值)
# 从银行贷款1000元，平均每年还pmt元，nper年还清，
# 年利率多少？
rate = np.rate(nper, pmt, 1000, 0)
print(round(rate, 2))
'''


'''
panda ： 对时间和二维数组(类似数据库的表,结构化数据)支持非常友好
'''


# print('编码练习')
# print(type('aa'.encode(encoding='unicode_escape')))
# print(type('aa'.encode(encoding='latin-1')))
# print('中国'.encode(encoding='utf8'))
# a='最高'.encode(encoding='gbk').decode('ISO-8859-1')
# print(a)
# b='aa'.encode(encoding='latin-1')
# c='最高'.encode(encoding='utf8')


# d='https://gitee.com/licuicui159/dataanalyze/blob/master/%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E6%80%BB.md'
# # import chardet
# # print(chardet.detect(d))
# # print(d.encode('utf8').decode('gbk'))
# from urllib.parse import unquote
# print(unquote(d, 'utf-8'))




