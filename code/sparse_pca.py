# 导入库
import math
import torch
import datetime
import warnings
import cvxpy as cp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn
from torch.nn.parameter import Parameter
from sklearn.linear_model import ElasticNet
from torch.utils.data import Dataset, DataLoader
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from torch.nn import Linear, ReLU, Sequential, Dropout
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 忽略警告，设置GPU
warnings.filterwarnings('ignore')
gpu = torch.device('cuda:0')


# 计算R2和RMSE
def r2_rmse(y_true, y_pred):
    r2 = r2_score(y_true, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
    for i in range(r2.shape[0]):
        print('{}: R2: {:.2f} RMSE: {:.4f}'.format(element[i], r2[i], rmse[i]))
    print('Averaged R2: {:.2f}'.format(np.mean(r2)))
    print('Averaged RMSE: {:.4f}'.format(np.mean(rmse)))
    return r2, rmse


# 计算邻接矩阵并标准化
def adjacency_matrix(X, mode='sc', epsilon=0.1, scale=0.4, l1=0.05, l2=0.5, self_con=0.2):
    x = X.cpu().numpy()
    if mode == 'rbf':
        k = RBF(length_scale=scale)
        A = k(x, x)
        A[np.abs(A) < epsilon] = 0
        D = np.diag(np.sum(A, axis=1) ** (-0.5))
        A = np.matmul(np.matmul(D, A), D)
    elif mode == 'pearson':
        A = np.corrcoef(x.T)
        A[np.abs(A) < epsilon] = 0
        D = np.diag(np.sum(A, axis=1) ** (-0.5))
        A = np.matmul(np.matmul(D, A), D)
    elif mode == 'sc':
        A = cp.Variable((x.shape[1], x.shape[1]))
        term1 = cp.norm(x * A - x, p='fro')
        term2 = cp.norm1(A)
        constraints = []
        for i in range(x.shape[1]):
            constraints.append(A[i, i] == 0)
            for j in range(x.shape[1]):
                constraints.append(A[i, j] >= 0)
        constraints.append(A == A.T)
        objective = cp.Minimize(term1 + l1 * term2)
        prob = cp.Problem(objective, constraints)
        prob.solve()
        A = A.value
        A = A + self_con * np.eye(x.shape[1])
        A[np.abs(A) < epsilon] = 0
        D = np.diag(np.sum(A, axis=1) ** (-0.5))
        A = np.matmul(np.matmul(D, A), D)
    elif mode == 'spca':
        k = x.shape[1]
        iter = 100
        u, d, v = np.linalg.svd(np.matmul(x.T, x))
        alpha = v[:k, :].T
        for i in range(iter):
            beta = np.zeros(alpha.shape)
            for j in range(k):
                net = ElasticNet(l1 + 2 * l2, l1 / (l1 + 2 * l2))
                net.fit(x, np.matmul(x, alpha[:, j]))
                beta_j = net.coef_
                beta[:, j] = beta_j.T
            u, d, v = np.linalg.svd(np.matmul(np.matmul(x.T, x), beta))
            alpha = np.matmul(u, v)
        A = np.matmul(beta, alpha.T)
        A[np.abs(A) < epsilon] = 0
        D = np.diag(np.sum(A, axis=1) ** (-1))
        A = np.matmul(D, A)
    sns.heatmap(A, annot=True)
    plt.show()
    A = torch.tensor(A, device=gpu)
    return A


# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index, :], self.label[index, :]

    def __len__(self):
        return self.data.shape[0]


# 自定义图卷积运算
class GraphConvolution(nn.Module):
    def __init__(self, n_input, n_output):
        super(GraphConvolution, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.weight = Parameter(torch.FloatTensor(n_input, n_output))
        self.reset_parameters()

    def forward(self, x, adj):
        temp = torch.matmul(x, self.weight)
        res = torch.matmul(adj.float(), temp.float())
        return res

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)


# 自定义多通道图卷积神经网络模型
class MCGCN(nn.Module):
    def __init__(self, n_variable, in_fc, gc, out_fc, n_output, direct_link=False, dropout=False):
        super(MCGCN, self).__init__()
        self.n_variable = n_variable
        self.n_in_fc = [n_variable] + list(in_fc)
        self.n_gc = [in_fc[-1]] + list(gc)
        self.n_out_fc = [gc[-1]] + list(out_fc)
        self.n_output = n_output
        self.dl = direct_link
        self.dropout = dropout
        self.act = ReLU()
        self.drop = Dropout(p=0.5)

        # 输入全连接层
        self.in_fc = nn.ModuleList()
        for i in range(len(in_fc)):
            temp = nn.ModuleList()
            for j in range(self.n_output):
                temp.append(Sequential(Linear(self.n_in_fc[i], self.n_in_fc[i + 1]), ReLU()))
            self.in_fc.append(temp)

        # 图卷积层
        self.gc = nn.ModuleList()
        for i in range(len(gc)):
            self.gc.append(GraphConvolution(self.n_gc[i], self.n_gc[i + 1]))

        # 输出全连接层
        self.out_fc = nn.ModuleList()
        for i in range(len(out_fc)):
            temp = nn.ModuleList()
            for j in range(self.n_output):
                temp.append(Sequential(Linear(self.n_out_fc[i], self.n_out_fc[i + 1]), ReLU()))
            self.out_fc.append(temp)

        # 输出层
        self.out = nn.ModuleList()
        for j in range(self.n_output):
            if self.dl:
                self.out.append(Linear(out_fc[-1] + n_variable, 1))
            else:
                self.out.append(Linear(out_fc[-1], 1))

    def forward(self, x, adj):
        feat_list = []

        # 输入全连接层
        for i in range(self.n_output):
            feat = x
            for fc in self.in_fc:
                feat = fc[i](feat)
            feat_list.append(feat)
        feat = torch.stack(feat_list, dim=1)

        # 图卷积层
        for gc in self.gc:
            feat = gc(feat, adj)
            feat = self.act(feat)

        # 输出全连接层
        res_list = []
        for i in range(self.n_output):
            res = feat[:, i, :]
            for fc in self.out_fc:
                res = fc[i](res)
            if self.dl:
                res = torch.cat((res, x), 1)
            if self.dropout:
                res = self.drop(res)
            res = self.out[i](res)
            res_list.append(res.squeeze())
        res = torch.stack(res_list, dim=1)
        return res


# 导入数据
data = pd.read_csv('data_preprocess.csv', index_col=0)
data.drop(['记录时间', '熔炼号', '钢种', '实际值-低碳锰铁', 'sb_record_time', 'sb_record_time_x', 'sb_record_time_y'], axis=1,
          inplace=True)
X = data.iloc[:, :-12]
y = data.iloc[:, -12:]
element = y.columns.map(lambda x: x[4:-2].capitalize())

# 数据集划分
seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
torch.manual_seed(seed)

# 数据标准化
scaler_X = StandardScaler().fit(X_train)
scaler_y1 = MinMaxScaler().fit(y_train)
scaler_y2 = StandardScaler().fit(y_train)
X_train_std = scaler_X.fit_transform(X_train)
X_test_std = scaler_X.fit_transform(X_test)
y_train_std = scaler_y1.fit_transform(y_train)
y_train_mm = scaler_y2.fit_transform(y_train)

# 数据移至GPU
X_train_gpu = torch.tensor(X_train_std, device=gpu, dtype=torch.float)
X_test_gpu = torch.tensor(X_test_std, device=gpu, dtype=torch.float)
y_train_gpu = torch.tensor(y_train_std, device=gpu, dtype=torch.float)

# 计算邻接矩阵
adj = adjacency_matrix(torch.tensor(y_train_mm, device=gpu), 'sc')

# 模型超参数设置以及生成
in_fc = (1024,)
gc = (256,)
out_fc = (256, 256)
mcgcn = MCGCN(X_train.shape[1], in_fc, gc, out_fc, y_train.shape[1]).to(gpu)

# 模型训练
t0 = datetime.datetime.now()
mcgcn.train()
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(mcgcn.parameters(), lr=0.001, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# 每个epoch
for epoch in range(200):
    loss = 0
    t1 = datetime.datetime.now()

    # 生成数据集和数据加载器
    data_train = MyDataset(X_train_gpu, y_train_gpu)
    dataloader = DataLoader(data_train, batch_size=64, shuffle=True)

    # 每个batch
    for item in dataloader:
        batch_X, batch_y = item[0], item[1]
        optimizer.zero_grad()

        # 计算训练误差并反向传播
        output_train = mcgcn(batch_X, adj)
        loss_train = criterion(output_train, batch_y)
        loss_train.backward()
        loss += loss_train.item()

        # 模型参数调整
        optimizer.step()
    scheduler.step()

    # 打印
    t2 = datetime.datetime.now()
    print('Epoch: {:03d} loss_train: {:.4f} time: {}'.format(epoch + 1, loss, t2 - t1))
t3 = datetime.datetime.now()
print('Optimization Finished! Time:', t3 - t0)

# 模型测试
mcgcn.eval()
y_pred = mcgcn(X_test_gpu, adj)
r2, rmse = r2_rmse(y_test, scaler_y1.inverse_transform(y_pred.cpu().detach().numpy()))
