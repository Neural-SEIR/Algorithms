#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/11 11:51
# @Author  : duoduo
# @File    : qyc.py
# 每一组8764个
# 这个程序搞温差负数总计50014个，分为十组，每组5000个，进行八组测试，两组验证误差
#并且自己预先处理好按照温差进行升序
#基于jieshu128

import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
datageshu=100


df = pd.read_excel('ceshishuju5ceshi.xlsx')
a = df['干球温度']
b = df['电力负荷']
c = df['温差']
listdianlifu = b[:5000]
listwenchafu = c[:5000]
xulie = []
print(listdianlifu[0])
for i in range(datageshu):
    xulie.append(i)
# 创建模型
linear = LinearRegression()
# 拟合模型
linear.fit(np.reshape(xulie, (-1, 1)), np.reshape(listwenchafu[:datageshu], (-1, 1)))
print(linear)
# 预测
y_pred = linear.predict(np.reshape(xulie, (-1, 1)))
plt.figure(figsize=(5, 5))  # 产生一个窗口
plt.scatter(xulie, listwenchafu[:datageshu])  # 画散点图
plt.plot(xulie, y_pred, color='red')

plt.show()

# 定义afa
# 第一种方法进行线性回归拟合,使用5000中间值作为5000个的α
def afa1(dao):
    zhi = abs(linear.coef_ * dao + linear.intercept_)
    # print(zhi)
    return zhi

# 第二种方法进行α用sin来拟合（0，1），用温差除以温差最大值乘以360当角度
wenchamax = listwenchafu[0] - listwenchafu[4999]
def afa2(dao):
    zhi = math.sin(listwenchafu[dao] / wenchamax * 360)
    return zhi
'''for i in range(100):
    b=afa2(i)
    print(b)'''
af = afa1(datageshu/2)
print(af)
# 多项式右侧关于（初始值）值的处理，12.8周五研究的,不能超过150，要不gamma值太大
def updata_fractional_accumulation(input_value, order):
    accumulation_value = np.zeros(len(input_value))
    for i in range(len(input_value)):
        for j in range(i + 1):
            tmp1 = math.gamma(i - j + order) / math.gamma(i - j + 1)
            tmp = tmp1 / math.gamma(order)
            accumulation_value[i] += tmp * input_value[j]
    return accumulation_value

d = updata_fractional_accumulation(listdianlifu[:datageshu], 0.1)
print(d)

N=datageshu
duoshaoi=3
jiyi=[]
def abc(b,c):
    # 进行方程编写，并且自己手写导入第一条数据，因为会有0为除数，所以自己要写入第一次循环
    #其中。左侧有一个N，需要替换的话是需要1 / af将1改成n，charuzhi2不需要
    charuzhi = (N / af * listdianlifu[0] - math.gamma(af) * (b*d[0])+c) / (N/af + math.gamma(af))
    print(charuzhi)
    charuzhi2 = (-charuzhi) / af
    # 将两个数作为一个
    tempabc = []
    tempabc.append(listdianlifu[0])
    tempabc.append(charuzhi)
    tempbiao1 = []
    #tempbiao1.append(0)
    tempbiao1.append(2)

    jiyi.append(charuzhi2)
    while(tempbiao1[0]!=datageshu):
        #print(len(tempabc))
        i=tempbiao1[0]

        #print('第' + str(i) + '次')
        #预测值
        print(i)
        #temp=(((1/af )*i*jiyi[0]+(1/af)*tempabc[i])*((1/(i+1))**(af)-(1/i)**(af))-(b*(tempabc[i]**(bt)))*math.gamma(af))/(((1/af)*(1/(i+1))**(af)-(1/i)**(af))+math.gamma(af))
        temp=(jiyi[0]+(i**(-af)-(i-1)**(-af))/i/af*tempabc[i-1]-math.gamma(af)/N*(b*d[i-1]+c))/((i**(-af)-(i-1)**(-af))/i/af+math.gamma(af)/N)
        #print(temp)
        print(temp)
        tempjia = ((tempabc[i-1]-temp) / i)*(i**(-af)-(i-1)**(-af))*(1/af)
        # 控制参数+1
        print(tempjia)
        i += 1
        jiyi[0] = jiyi[0] + tempjia-4*math.log(i)
        #jiyi[0] = jiyi[0] + tempjia - math.e**(i/20.5)
        tempbiao1[0]=i
        tempabc.append(temp)
    #print(tempabc)
    #在方法中进行误差分析，希望得出的是最小的误差
    wucha=[]
    fanhui=[]
    wucha.append(0)
    for i in range(datageshu):
        zhi=abs(tempabc[i]-listdianlifu[i])/listdianlifu[i]
        wucha[0]=wucha[0]+zhi
    result=wucha[0]/datageshu
    fanhui.append(result)
    fanhui.append(b)
    fanhui.append(c)
    #print(tempabc)
    #return fanhui
    return tempabc

panduan=[]
panduan.append(50)
zuizhongwucha=[]

#执行函数
'''for i in range(-10,10,1):
    i=i/10
    for j in range(-40000,40000,100):
        cd=abc(i,j)
        if cd[0]<panduan[0]:
            zuizhongwucha=cd
             continue
print(zuizhongwucha)'''
cd=abc(-0.95,-5000)
print(cd)
xulie=[]
sh=[]
for i in range(datageshu):
    xulie.append(i)
    sh.append(int(cd[i]))
print(len(xulie))
print(len(sh))
plt.plot(xulie, sh,'-')
plt.show()

wuchazhi=[]
wuchasum=[]
wuchasum.append(0)
for i in range(datageshu):
    zhi=(abs(sh[i]-listdianlifu[i]))/listdianlifu[i]
    wuchazhi.append(zhi)
    wuchasum[0]=wuchasum[0]+zhi
print(wuchasum[0]/datageshu)


#推25个,100推25
yucezhi = []
yuce =list(listdianlifu[:datageshu])
yuce.append(float(cd[99]))
print("以下是jiyi【0】")
print(jiyi[0])
print(yuce[100])
b=-0.95
c=-5000
for i in range(100,125):
    d1 = updata_fractional_accumulation(yuce, 0.1)
    print(d1[i])
    print(i)

    temp=(jiyi[0]+(i**(-af)-(i-1)**(-af))/i/af*yuce[i]-math.gamma(af)/i*(b*d1[i]+c))/((i**(-af)-(i-1)**(-af))/i/af+math.gamma(af)/i)
    print(temp)
    yucezhi.append(temp)
    yuce.append(temp)
    tempjia = ((yuce[i]-temp) / (i+1)/af)*(i**(-af)-(i-1)**(-af))
    #tempjia = ((yuce[i-1]-temp) / i)*(i**(-af)-(i-1)**(-af))*(1/af)
    print(tempjia)
    jiyi[0] = jiyi[0] + tempjia - 4 * math.log(i)
    print("jiyi[0]")
    print(jiyi[0])

print(yucezhi)
print(yuce)



'''zhi = []
zhi.append(cd[99])

    # 进行方程编写，并且自己手写导入第一条数据，因为会有0为除数，所以自己要写入第一次循环
    # 其中。左侧有一个N，需要替换的话是需要1 / af将1改成n，charuzhi2不需要
    charuzhi = (N / af * listdianlifu[0] - math.gamma(af) * (b * d[0]) + c) / (N / af + math.gamma(af))
    charuzhi2 = (-charuzhi) / af - duoshaoi
    # 将两个数作为一个
    tempabc = []
    tempabc.append(listdianlifu[0])
    tempabc.append(charuzhi)
    tempbiao1 = []
    # tempbiao1.append(0)
    tempbiao1.append(2)
    jiyi = []
    jiyi.append(charuzhi2)
    while (tempbiao1[0] != datageshu):
        # print(len(tempabc))
        i = tempbiao1[0]
        # print('第' + str(i) + '次')
        # 预测值
        # temp=(((1/af )*i*jiyi[0]+(1/af)*tempabc[i])*((1/(i+1))**(af)-(1/i)**(af))-(b*(tempabc[i]**(bt)))*math.gamma(af))/(((1/af)*(1/(i+1))**(af)-(1/i)**(af))+math.gamma(af))
        temp = (jiyi[0] + (i ** (-af) - (i - 1) ** (-af)) / i / af - N / math.gamma(af) * (b * d[i - 1] + c)) / (
                    (i ** (-af) - (i - 1) ** (-af)) / i / af + N / math.gamma(af))
        # print(temp)
        tempjia = ((jiyi[0] - temp) / i) * (i ** (-af) - (i - 1) ** (-af)) * (1 / af)
        # 控制参数+1
        i += 1
        jiyi[0] = jiyi[0] + tempjia - datageshu / 3 * math.log(i)
        # jiyi[0] = jiyi[0] + tempjia - math.e**(i/20.5)
        tempbiao1[0] = i
        tempabc.append(temp)'''





'''for i in range(-10,10,1):
    i=i/10
    print(i)
    cd=abc(i,0)
    if cd[0]<panduan[0]:
        zuizhongwucha=cd
    else:
        continue
print(zuizhongwucha)'''