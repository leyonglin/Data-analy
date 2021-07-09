# import typing_extensions
# import numpy as np
# import matplotlib.pyplot as mp
# import matplotlib.gridspec as mg
# print('绘制第一个窗口')
# mp.figure('A Figure',facecolor='gray')
# x=np.array([0,1,2,3,4,5])
# y=np.array([0,5,4,3,2,10])
# print('基本图形')
# mp.plot(x,y)
# print('绘制水平线，垂直线（分段函数）')
# mp.hlines([1,2],1,2)
# mp.vlines(2,2,3)
# # mp.plot(x,y)
# print('正弦函数')
# x=np.linspace(-np.pi,np.pi,1000)
# sinx=np.sin(x)
# cosx=np.cos(x)
# # mp.plot(x,y)
# print('设置坐标轴范围')
# # mp.xlim(0,np.pi)
# # mp.ylim(0,1.5)
# print('设置坐标的刻度')
# vals=[-np.pi,-np.pi/2,0,np.pi/2,np.pi]
# # texts=['-Π','-Π/2','0','Π/2','-Π']
# # #latex语法
# xtexts=[r'$-\pi$',r'$-\frac{\pi}{2}$','0',r'$\frac{\pi}{2}$',r'$\pi$']
# mp.xticks(vals,xtexts)
# mp.yticks([-1.0,-0.5,0.5,-1.0])
# print('设置坐标轴位置')
# ax=mp.gca()获取当前坐标轴字典，{'left':左轴,'right':右轴,'bottom':下轴,'top':上轴 }
# axis=ax.spines['坐标轴名']获取某个坐标轴
# 设置坐标轴的位置。 该方法需要传入2个元素的元组作为参数
# type: <str> 移动坐标轴的参照类型  一般为'data' (以数据的值作为移动参照值)
# val:  参照值
# axis.set_position((type, val))
# 设置坐标轴的颜色
# color: <str> 颜色值字符串
# #去掉上右两个轴，把下左两轴移到中间
# ax=mp.gca()
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
# ax.spines['left'].set_position(('data',0))
# ax.spines['bottom'].set_position(('data',0))

# print('设置曲线格式')
# # linestyle：线型 '-' '--' '-.' ':' '.'
# # linewhidth: 线宽  倍数数值
# # color: 颜色 英语单词或简写或#495434 或 (1,1,1) (1,1,1,1)  #四维元组最后一位表示为透明度
# # alpha:透明度
# # label:设置图例的时候需要用到
# mp.plot(x,sinx,linestyle='--',linewidth=2,color='orangered',alpha=0.8,label=r'$y=sin(x)$')
# mp.plot(x,cosx,linestyle='--',linewidth=2,color='blue',alpha=0.8,label=r'$y=\frac{1}{2}cos(x)$')
# # print(help(mp.plot))
# print('设置图例')
# '''            ===============   =============
#             Location String   Location Code
#             ===============   =============
#             'best'            0
#             'upper right'     1
#             'upper left'      2
#             'lower left'      3
#             'lower right'     4
#             'right'           5
#             'center left'     6
#             'center right'    7
#             'lower center'    8
#             'upper center'    9
#             'center'          10
#             ===============   =============
#             help(mp.legend)中有'''
# mp.legend()
# print('绘制特殊点')
# '''
# mp.scatter(xarray,yarray,    #x和y的坐标数值
# marker='',                   #特殊点的形状
# s='',                        #大小
# edgecolor='',                #边缘色
# facecolor='',                #填充色
# zorder=3                     #图层
# )'''
# pointx=[np.pi/2,np.pi/2]
# pointy=[1,0]
# mp.scatter(pointx,pointy,
# marker='o',s=70,color='red',label='simple point',zorder=3)
# print('在图标中为某个点添加备注')
# mp.annotate(
#     r'$[\frac{\pi}{2},1]$',       # 备注文本信息
#     xycoords='data',              # 数据坐标轴，即使用当前坐标
#     xy=(np.pi/2,1),               # 原坐标特殊点位置
#     textcoords='offset points',   # 备注坐标类型，选择偏移型就可以
#     xytext=(50,30),               # 备注位置相对原坐标点偏移量
#     fontsize=14,                  #
#     arrowprops=dict(
#         arrowstyle='-|>',
#         connectionstyle='angle3'  #连接线样式
#     )
# )
# mp.legend()
# # 绘制第二个窗口
# '''
# mp.figure(
# '',               #窗口标题，一定要有
# figsize=(4,3),    #窗口大小
# dpi=120,          #像素密度，可不配置
# facecolor=''      #背景颜色
# )
# '''
# mp.figure('B Figure',facecolor='blue')
# mp.plot([1,2],[2,1])
# mp.title('B title',fontsize=12)  #坐标轴标题
# mp.xlabel('price',fontsize=12)  #x轴显示为什么
# mp.ylabel('time',fontsize=12)  #y轴显示为什么
# mp.tick_params(labelsize=12)     #坐标轴的刻度大小
# # -     ：粗线
# # --    ：虚线
# # -.    ：点虚线
# # ：    ：点线
# mp.grid(linestyle='--')               #设置图标网格线
# mp.tight_layout()                   #设置紧凑布局，把图标相关参数都显示在窗口中
# # 再绘制第一个窗口，标题存在则不会重新创建
# mp.figure('A Figure',facecolor='lightgray')
# mp.show()

# print('子图')
# print('矩阵式布局')
# mp.figure('Subplot',facecolor='lightgray')
# for i in range(1,10):
#     mp.subplot(3,3,i)
#     # 内容为一个文本
#     mp.text(0.5,0.5,i,ha='center',va='center',size=36,alpha=0.6)
#     mp.xticks([])
#     mp.yticks([])
#     mp.tight_layout()
# print('网格式布局，支持矩阵子图空间合并')
# mp.figure('grid layout',facecolor='lightgray')
# gs=mg.GridSpec(3,3)
# # mp.subplot(gs[1,1])     #操作就是切片
# mp.subplot(gs[0,:2])
# # mp.subplot(gs[:2,2])
# mp.text(0.5,0.5,1,ha='center',va='center',size=36,alpha=0.6)
# mp.xticks([])
# mp.yticks([])
# mp.tight_layout()
# print('自由式布局')
# '''
# mp.axes([left_buttom_x,left_buttom_x,width,height])
# 原点=左下角x轴的坐标(left_buttom_x),左下角y轴的坐标(left_buttom_x)
# width: 宽度
# height：高度
# '''
# mp.axes([0.03,0.5,0.94,0.4])
# mp.text(0.5,0.5,1,ha='center',va='center',size=36,alpha=0.6)
# mp.axes([0.03,0.03,0.54,0.4])
# mp.text(0.5,0.5,2,ha='center',va='center',size=36,alpha=0.6)
# mp.show()

# print('刻度定位器')
# mp.figure('locators')
# #设置x轴的可视范围
# mp.xlim(1,10)
# #刻度定位器(还有很多其它的定位器Locator)
# ax=mp.gca()
# ax.spines['top'].set_color('none')
# ax.spines['left'].set_color('none')
# ax.spines['right'].set_color('none')
# ax.spines['bottom'].set_position(('data',0.5))
# mp.yticks([])
# ax.xaxis.set_major_locator(mp.MultipleLocator(1))
# ax.xaxis.set_minor_locator(mp.MultipleLocator(0.1))
# mp.show()

# print('多刻度定位器')
# locators=['mp.NullLocator()',
# 'mp.MaxNLocator(nbins=4)',
# 'mp.FixedLocator([3,6,9])',
# 'mp.AutoLocator()']    #刻度定位器(还有很多其它的定位器Locator)
# mp.figure('locators',facecolor='lightgray')
# for i,locator in enumerate(locators):
#     mp.subplot(len(locators),1,i+1)
#     mp.xlim(1,10)
#     ax=mp.gca()
#     ax.spines['top'].set_color('none')
#     ax.spines['left'].set_color('none')
#     ax.spines['right'].set_color('none')
#     ax.spines['bottom'].set_position(('data',0.5))
#     mp.yticks([])
#     loc=eval(locator)
#     # ax.xaxis.set_major_locator(mp.MultipleLocator(1))
#     ax.xaxis.set_major_locator(loc)
#     ax.xaxis.set_minor_locator(mp.MultipleLocator(0.1))
# mp.show()

# print('刻度网格线')
# mp.figure('grid line',facecolor='lightgray')
# ax=mp.gca()
# '''参数
# ax.grid(
#     which='',    #'major'/'minor'/both  主刻度和次刻度
#     axis='',     #x/y/both
#     linewidth=1, linestyle='', color='',alpha=0.5     
# )'''
# ax.grid(which='major',axis='both',color='orangered',alpha=0.75)
# ax.grid(which='minor',axis='both',color='orangered',alpha=0.25)
# #修改刻度定位器
# ax.xaxis.set_major_locator(mp.MultipleLocator(1))
# ax.xaxis.set_minor_locator(mp.MultipleLocator(0.1))

# ax.yaxis.set_major_locator(mp.MultipleLocator(250))
# ax.yaxis.set_minor_locator(mp.MultipleLocator(50))

# #绘制曲线
# y=[1,10,100,1000,100,10,1]
# #'o-'表示将点用圈表示出来，正常绘图
# # mp.plot(y,'o-',color='dodgerblue')
# #半对数坐标绘图
# mp.semilogy(y,'o-',color='dodgerblue')
# mp.show()

# print('散点图,就是特殊点')
# #生成符合正态分布的随机数
# '''np.random.normal(期望值,标准差,数量)，3倍标准差涵盖了99.7%'''
# x=np.random.normal(175,20,300)
# y=np.random.normal(70,10,300)
# mp.figure('persons',facecolor='lightgray')
# mp.title('persons',fontsize=18)
# mp.xlabel('x',fontsize=14)
# mp.ylabel('y',fontsize=14)
# mp.grid(linestyle=':')
# # mp.scatter(x,y,marker='o',s=70,label='persons',color='dodgerblue')
# d=(x-175)**2+(y-70)**2
# # cmap颜色渐变---jet ， jet_r 翻转
# mp.scatter(x,y,marker='o',s=70,label='persons',c=d,cmap='jet')
# mp.legend()
# mp.show()

# print('图像对象之填充')
# x=np.linspace(0,8*np.pi,1000)
# sinx=np.sin(x)
# cosx=np.cos(x/2)/2
# mp.figure('fill',facecolor='lightgray')
# mp.title('fill',fontsize=18)
# mp.grid(linestyle=':')
# mp.plot(x,sinx,color='dodgerblue',
# label=r'$y=sinx(x)$')
# mp.plot(x,cosx,color='orangered',
# label=r'$y=\frac{1}{2}cos(\frac{x}{2}$)')
# mp.fill_between(x,sinx,cosx,sinx>cosx,color='dodgerblue',alpha=0.3)
# mp.fill_between(x,sinx,cosx,sinx<cosx,color='orangered',alpha=0.3)
# mp.legend()
# mp.show()

# print('图像对象之条形图')
# apples = np.array([15,95,34,58,76,12,59,81,42,36,17,95])
# oranges=np.array([65,48,92,13,58,76,45,91,35,62,46,85])
# mp.figure('bar',facecolor='lightgray')
# mp.title('bar chart',fontsize=18)
# mp.grid(linestyle=':')
# x=np.arange(apples.size)
# '''mp.bar(x,apples,10,0.8,color='limegreen',label='Apples',align='edge')
# x:x轴
# apples：y轴
# 10: 离x轴的距离，可以是一个数组，可选
# 0.8 ：0~1取值，柱子的宽度
# '''
# mp.bar(x-0.2,apples,0.4,color='limegreen',label='Apples',align='center')
# mp.bar(x+0.2,oranges,0.4,color='orangered',label='Oranges',align='center')
# mp.xticks(x,["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sept","Oct","Nov","Dec"])
# mp.legend()
# mp.show()

# print('图像对象之饼图')
# '''
# mp.pie(
#     values,      #值列表
#     spaces,      #扇形之间的间距列表
#     labels,      #标签列表
#     colors,      #颜色列表
#     '%d%%',      #标签所占比例格式
#     shadow=True, #是否显示阴影
#     startangle=90, #逆时针绘制饼状图时的起始角度
#     radius=1      #半径
# )
# '''
# mp.figure('pip',facecolor='lightgray')
# mp.title('pie',fontsize=20)
# values=[26,17,21,29,11]
# spaces=[0.05,0.01,0.01,0.01,0.01]
# labels=['python','javascript','c++','java','php']
# colors=['dodgerblue','orangered','limegreen','violet','gold']
# #等轴比例(x轴与y轴等比例输出)
# mp.axis('equal')
# mp.pie(values,spaces,labels,colors,'%.1f%%',shadow=True)
# mp.legend()
# mp.show()

# print('图像对象之等高线图')
# '''mp.contourf(x,y,z,
#               8,              #分为几分(颜色)
#               cmap='jet')

# mp.contourf(
#     x,          #网格坐标矩阵的x坐标
#     y,          #网格坐标矩阵的y坐标
#     z,          #网格坐标矩阵的z坐标
#     8,          #把等高线绘制成8部分
#     color='block, #等高线的颜色
#     linewidths=0.5 #线宽 
#     )
# '''
# #使用api和方程生成模拟数据
# n=100
# x,y=np.meshgrid(np.linspace(-3,3,n),
#                 np.linspace(-3,3,n))
# z=(1-x/2+x**5+y**3)*\
#     np.exp(-x**2-y**2)
# mp.figure('contour',facecolor='lightgray')
# mp.title('contour',fontsize=16)
# mp.grid(linestyle=":")
# cntr=mp.contour(x,y,z,8,colors='black',linewidths=0.5)
# #设置等高线上的高度标签文本,可选
# mp.clabel(cntr,fmt='%.2f',inline_spacing=2,fontsize=8)
# # 填充色，可选
# mp.contourf(x,y,z,8,cmap='jet')
# mp.show()

# print('图像对象之热成像图')
# n=100
# x,y=np.meshgrid(np.linspace(-3,3,n),
#                 np.linspace(-3,3,n))
# z=(1-x/2+x**5+y**3)*\
#     np.exp(-x**2-y**2)
# # 把矩阵图形化,origin:坐标轴方向(upper:缺省值，左上角，lower：左下角)
# mp.imshow(z,cmap='jet',origin='lower')
# #使用颜色条显示热度值，可选
# mp.colorbar()
# mp.show()

# print('图像对象之3d绘图')
# from mpl_toolkits.mplot3d import axes3d
'''
api
ax3d.scatter(  #绘制三维点阵
    x,              #x轴坐标数组
    y,              #y轴坐标数组
    z,              #z轴坐标数组
    marker='',      #点型
    s=10,           #大小
    zorder='',      #图层序号
    color='',       #颜色
    edgecolor='',   #边缘颜色
    facecolor='',   #填充色
    c=v,            #颜色值，根据cmap映射应用相应颜色
    cmap=''
)         
ax3d.plot_surface(  #绘制三维曲面
    x,              #x轴坐标数组
    y,              #y轴坐标数组
    z,              #z轴坐标数组
    rstride=30,     #行跨距,跨距其实决定像素细腻度
    cstride=30,     #列跨距
    cmap='jet'      #颜色映射
)    
ax3d.plot_wireframe( #绘制三维线框图
    x,              #x轴坐标数组
    y,              #y轴坐标数组
    z,              #z轴坐标数组
    rstride=30,     #行跨距,跨距其实决定像素细腻度
    cstride=30,     #列跨距    
    linewidth=1,    #线宽
    color=''
)  
'''
# print('三维点阵')
# n=300
# x=np.random.normal(0,1,n)
# y=np.random.normal(0,1,n)
# z=np.random.normal(0,1,n)
# mp.figure('3d scatter',facecolor='lightgray')
# ax3d = mp.gca(projection='3d')
# ax3d.set_xlabel('x')
# ax3d.set_ylabel('y')
# ax3d.set_zlabel('z')
# d=x**2+y**2+z**2
# ax3d.scatter(x,y,z,s=60,marker='o',c=d,cmap='jet')
# print('三维曲面')
# n=100
# x,y=np.meshgrid(np.linspace(-3,3,n),
#                 np.linspace(-3,3,n))
# z=(1-x/2+x**5+y**3)*\
#     np.exp(-x**2-y**2)
# mp.figure('3d surface',facecolor='lightgray')
# ax3d = mp.gca(projection='3d')
# ax3d.set_xlabel('x')
# ax3d.set_ylabel('y')
# ax3d.set_zlabel('z')
# # ax3d.plot_surface(x,y,z,cstride=30,rstride=30,cmap='jet')
# print('三维线框图')
# ax3d.plot_wireframe(x,y,z,cstride=30,rstride=30,linewidth=1,color='dodgerblue')
# mp.show()

# print('极坐标系')
# t=np.linspace(0,4*np.pi,1000)
# r=0.8*t
# #描述极径和极角(有序数对（ρ，θ）)的线性关系，适合显示与角度有关的图像(定位和导航)
# mp.figure('polar',facecolor='lightgray')
# # mp.gca(projection='polar') 这句话是关键
# mp.gca(projection='polar')
# mp.title('polar',fontsize=20)
# mp.xlabel(r'$\theta$',fontsize=14)
# mp.ylabel(r'$\rho$',fontsize=14)
# mp.tick_params(labelsize=10)
# mp.grid(linestyle=":")
# mp.plot(t,r)
# mp.show()

# print('简单动画1')    #在一段时间内快速连续的重新绘制图像的过程
# import matplotlib.animation as ma
# '''
# 定义更新函数行为
# def update(number):   #计数器，number是自动传入的参数，从0开始递增1
#     pass
# 每隔10毫秒执行一次update更新函数，作用于mp.gcf()当前窗口对象
# mp.gcf():获取当前窗口
# update：更新函数
# interval：间隔时间(单位：毫秒)
# anim = ma.FuncAnimation(mp.gcf(),update,interval=10)
# '''
# #随机生成100个点对象
# n=100
# balls=np.zeros(100,dtype=[
#     ('position',float,2),
#     ('size',float,1),
#     ('growth',float,1),
#     ('color',float,4)
# ])
# # for ball in balls:
# #     print(ball)     #100个([0., 0.], 0., 0., [0., 0., 0., 0.])
# #初始化balls数组每个字段的属性值
# # 均匀分布np.random.uniform(0,1,(n,2)) # 0~1 均匀生成n行2列
# balls['position']=np.random.uniform(0,1,(n,2))
# balls['size']=np.random.uniform(50,70,n)
# balls['growth']=np.random.uniform(10,20,n)
# balls['color']=np.random.uniform(0,1,(n,4))
# # for ball in balls:
# #     print(ball) 
# mp.figure('animation',facecolor='lightgray')
# mp.title('animation',fontsize=14)
# mp.xticks([])
# mp.yticks([])
# sc = mp.scatter(balls['position'][:,0],
# balls['position'][:,1],s=balls['size'],color=balls['color']
# )
# #动起来
# def update(number):
#     # print(number)  #内置变量
#     # 重新修改index位置元素的属性值（消失）
#     index = number % 100
#     balls['position'][index] = \
#         np.random.uniform(0,1,(1,2))
#     balls['size'][index]=np.random.uniform(50,70,1)
#     balls['size'] += balls['growth']
#     #重新绘制界面
#     sc.set_sizes(balls['size'])
#     sc.set_offsets(balls['position'])


# anim = ma.FuncAnimation(mp.gcf(),update,interval=30)
# mp.show()

# print('简单动画2') 
# '''
# def update(data):
#     t,v=data
#     ...
#     pass
# def generator():
#     yield t,v
# # 每隔10毫秒将会先调用生成器，获取生成器返回的数据
# # 把生成器返回的数据交给并且调用update函数，执行更新图像函数
# anim = ma.FuncAnimation(mp.gcf(),update,generator,interval=10)
# '''
# import matplotlib.animation as ma
# mp.figure("Signal", facecolor='lightgray')
# mp.title("Signal", fontsize=14)
# mp.xlim(0, 10)
# mp.ylim(-3, 3)
# mp.grid(linestyle='--', color='lightgray', alpha=0.5)
# pl = mp.plot([], [], color='dodgerblue', label='Signal')[0]
# pl.set_data([],[])
# x = 0
# def update(data):
# 	t, v = data
# 	x, y = pl.get_data()
# 	x.append(t)
# 	y.append(v)
# 	#重新设置数据源
# 	pl.set_data(x, y)
# 	#移动坐标轴
# 	if(x[-1]>10):
# 		mp.xlim(x[-1]-10, x[-1])
# def y_generator():
# 	global x
# 	y = np.sin(2 * np.pi * x) * np.exp(np.sin(0.2 * np.pi * x))
# 	yield (x, y)
# 	x += 0.05
# 	print(x)

# anim = ma.FuncAnimation(mp.gcf(), update, y_generator, interval=20)
# mp.tight_layout()
# mp.show()

#生成器
class TestYield:
    def gen_iterator(self):
        for j in range(3):
            print(f"do_something-{j}")
            # yield在for循环内部
            yield j
            print("aaa")

    def call_gen_iterator(self):
        # yield并不是直接返回[0,1,2]，执行下边这句后result_list什么值都没有
        # print("bbb")
        result_list = self.gen_iterator()
        # i每请求一个数据，才会触发gen_iterator生成一个数据
        for i in result_list:
            print(f"call_gen_iterator-{i}")
            print("ccc")
            
        # print("ddd")

if __name__ == "__main__":
    obj = TestYield()
    obj.call_gen_iterator()

#推导式
'''
rise = num1 > num2
color = [ 'white' if x else 'green' for x in rise ]

color = np.zeros(rise.size,dtype='U5)
color = 'green'
color[rise] = 'white'
'''
'''
# numpy的掩码运算(获取总体数据的子集),对周一到周五分别进行数据汇总
# ave_prices = np.zeros(5)
# for wday in range(ave_closing_prices.size):
#     ave_closing_prices[wday] = closing_prices[wdays == wday].mean()
'''

'''
#绘制影线(k线图)
# mp.vlines(dates,lowest_prices,highest_prices,color='white')
'''




