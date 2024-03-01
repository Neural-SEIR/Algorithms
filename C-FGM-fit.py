#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/13 13:07
# @Author  : duoduo
# @File    : model.py
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
#注意注意注意，有注释，执行函数在88行，进行rou和c的调试，测试函数在最后,如果不改变循环的范围，只需要改文件名称和19行组别
#定义个数
zui=[]
for xunhuan in range(1,131):
    datageshu=96
    zubie=xunhuan
    print(zubie)
    df = pd.read_excel('six.xlsx')
    a = df['Temp']
    b = df['Load']
    listdianlifu = []
    zhi = b[(zubie - 1) * datageshu:zubie * datageshu]
    #print(zhi)
    for i in range(datageshu):
        temp=zhi[(zubie-1)*datageshu+i]
        listdianlifu.append(temp)
    #print(listdianlifu)

    #定义af
    af=0.6
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
    #print(d)

    N=datageshu

    def abc(b,c):
        jiyi = []
        #print(str(b)+" "+str(c))
        # 进行方程编写，并且自己手写导入第一条数据，因为会有0为除数，所以自己要写入第一次循环
        #其中。左侧有一个N，需要替换的话是需要1 / af将1改成n，charuzhi2不需要
        charuzhi = (N / af * listdianlifu[0] - math.gamma(af) * (b*d[0])+c) / (N/af + math.gamma(af))
        charuzhi2 = (-charuzhi) / af
        # 将两个数作为一个
        tempabc = []
        tempabc.append(listdianlifu[0])
        tempabc.append(charuzhi)
        tempbiao1 = []
        tempbiao1.append(2)
        jiyi.append(charuzhi2)
        #误差返回


        while(tempbiao1[0]!=datageshu):
            i=tempbiao1[0]
            #预测值
            temp=(jiyi[0]+(i**(-af)-(i-1)**(-af))/i/af*tempabc[i-1]-math.gamma(af)/N*(b*d[i-1]+c))/((i**(-af)-(i-1)**(-af))/i/af+math.gamma(af)/N)
            tempjia = ((tempabc[i-1]-temp) / i)*(i**(-af)-(i-1)**(-af))*(1/af)
            # 控制参数+1
            i += 1
            jiyi[0] = jiyi[0] + tempjia
            tempbiao1[0]=i
            tempabc.append(temp)
            #在方法中进行误差分析，希望得出的是最小的误差
        #print(tempabc)
        wucha = []
        fanhui = []
        wucha.append(0)
        for i in range(datageshu):
            zhi=abs(tempabc[i]-listdianlifu[i])/listdianlifu[i]
            wucha[0]=wucha[0]+zhi
        aba=wucha[0]
        result=aba/datageshu
        fanhui.append(result)
        fanhui.append(b)
        fanhui.append(c)
        return fanhui


    panduan=[]
    panduan.append(50)
    zuizhongwucha=[]

    #执行函数
    minwucha=[]
    minwucha.append(0.3)
    minwucha.append(0)
    minwucha.append(0)
    for i in range(-99,99,10):
        i=i/100
        for j in range(-500000,500000,100):
            cd=abc(i,j)

            if(cd[0]<minwucha[0]):
                minwucha[0]=cd[0]
                minwucha[1]=cd[1]
                minwucha[2]=cd[2]
            else:
                continue

    '''cd=abc(0.01,-394100)
    print(cd)'''


    #以下为检查测试，单rou和单c进行验证
    '''cd=abc(-0.8,-430000)
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
    print(wuchasum[0]/datageshu)'''
    print(minwucha)
    zui.append(minwucha)
print(zui)