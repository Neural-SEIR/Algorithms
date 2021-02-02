import torch 
import math
import pandas as pd
import numpy as np
import xlrd
from xlutils.copy import copy

def nn_optimize(x,y):

    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)

    model = torch.nn.Sequential(
        torch.nn.Linear(3,1),
        torch.nn.Flatten(0,1)
    )

    loss_fn = torch.nn.MSELoss('reduction=sum')

    learning_rate = 1e-3

    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    for t in range(20000):
        y_pred = model(xx)

        loss = loss_fn(y_pred, y)

        if t % 100 == 99:
            print(t, loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    linear_layer = model[0]

    print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')

    return ([linear_layer.weight[:, 2].item(),linear_layer.weight[:, 1].item(),linear_layer.weight[:, 0].item(),linear_layer.bias.item()])
    

# torch.set_default_tensor_type(torch.FloatTensor)

# read in data from csv "New York State 1101-1201"

infection_new = pd.read_excel('New York State 0401--0201.xlsx')

date = infection_new['date']

#print(date)

ydata = infection_new['positiveIncrease']

zdata = infection_new['recoveredIncrease']

# y1 = torch.tensor(y)
# y2 = y1.float()
# print(len(y1))
# x = torch.linspace(0, len(y1)-1, len(y1))

# print(len(x))

# z1 = torch.tensor(z)
# z2 = z1.float()

# positive = nn_optimize(x,y2)

# print(positive[0])

# print(positive[1])

# recovered = nn_optimize(x,z2)

# print(recovered[0])

# print(recovered[1])


for i in range(0,len(ydata)-61):
    #y  = list.index(ydata, i, 60 + i)

    y = ydata[i : 61 + i]

    y = np.array(y)

    z = zdata[i : 61 + i]

    z = np.array(z)

    y1 = torch.tensor(y)

    y2 = y1.float()

    x = torch.linspace(0, 60, 61)

    z1 = torch.tensor(z)
    
    z2 = z1.float()

    positive = nn_optimize(x,y2)

    print(positive)

    recovered = nn_optimize(x,z2)

    print(recovered)

    result = positive + recovered

    print(result)

    filename = 'test.xls'# 文件名
    rb = xlrd.open_workbook(filename, formatting_info=True)  
    # formatting_info=True: 保留原数据格式
    wb = copy(rb) 			# 复制页面 
    ws = wb.get_sheet(0) 	# 取第一个sheet 
    # ----- 按(row, col, str)写入需要写的内容 -------
    ws.write(i+1, 0, str(date[61 + i]))
    for j in range(len(result)):
        ws.write(i+1, 1 + j, result[j])  
    # ----- 按(row, col, str)写入需要写的内容 -------
    wb.save(filename) 		# 保存文件

# temp = pd.DataFrame(result)

# write = pd.ExcelWriter('1.xlsx')

# temp.to_excel(write,'Sheet1')

# write.save()