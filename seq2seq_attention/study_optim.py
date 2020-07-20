import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


x_train = torch.from_numpy(x_train)

y_train = torch.from_numpy(y_train)


# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(1, 5)  # input and output is 1 dimension
        self.linear2 = nn.Linear(5,1)
    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        return out


model = LinearRegression()
print(model.linear1)
# 微调：自定义每一层的学习率

# 定义loss和优化函数
criterion = nn.MSELoss()
optimizer = optim.SGD(
                [{"params":model.linear1.parameters(),"lr":0.01},
                 {"params":model.linear2.parameters()}],
                      lr=0.02)
lambda1 = lambda epoch: epoch//100
lambda2 = lambda epoch: 0.95**epoch

step_schedule = optim.lr_scheduler.StepLR(step_size=20,gamma=0.9,optimizer=optimizer)
cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=20,eta_min=0.0004)
exponent_schedule = optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.9)
reduce_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
multi_schedule = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[120,180])
lambda_schedule = optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=[lambda1,lambda2])

step_lr_list = []
cosine_lr_list = []
exponent_lr_list = []
reduce_lr_list = []
loss_list = []
multi_list = []
lambda1_list = []
lambda2_list = []
# 开始训练
num_epochs = 240
for epoch in range(num_epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)

    # forward
    out = model(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    step_schedule.step()
    cosine_schedule.step()
    exponent_schedule.step()
    reduce_schedule.step(loss)
    multi_schedule.step()
    lambda_schedule.step()

    # print(schedule.get_lr())
    loss_list.append(loss.item())
    # print(optimizer.param_groups[0]["lr"])
    step_lr_list.append(step_schedule.get_lr()[0])
    cosine_lr_list.append(cosine_schedule.get_lr()[0])
    exponent_lr_list.append(exponent_schedule.get_lr()[0])
    reduce_lr_list.append(optimizer.param_groups[0]["lr"])
    multi_list.append(multi_schedule.get_lr()[0])
    lambda1_list.append(optimizer.param_groups[0]["lr"])
    lambda2_list.append(optimizer.param_groups[1]["lr"])
# print(optimizer.param_groups[0]["lr"])
#     print(optimizer.param_groups[1]["lr"])

    # if (epoch+1) % 20 == 0:
    #     print('Epoch[{}/{}], loss: {:.6f}'
    #           .format(epoch+1, num_epochs, loss.item()))
plt.subplot(121)
plt.plot(range(len(loss_list)),loss_list,label="loss")
plt.legend()
plt.subplot(122)
plt.plot(range(len(step_lr_list)),step_lr_list,label="step_lr")
plt.plot(range(len(cosine_lr_list)),cosine_lr_list,label="cosine_lr")
plt.plot(range(len(exponent_lr_list)),exponent_lr_list,label="exponent_lr")
plt.plot(range(len(reduce_lr_list)),reduce_lr_list,label="reduce_lr")
plt.plot(range(len(multi_list)),multi_list,label="multi_lr")
plt.plot(range(len(lambda1_list)),lambda1_list,label="lambda1_lr")
plt.plot(range(len(lambda2_list)),lambda2_list,label="lambda2_lr")
plt.legend()
plt.show()
