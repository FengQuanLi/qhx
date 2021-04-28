import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
# distributions概率分布和采样函数
import torch
import torch.nn as nn
from Layers import DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayers import Norm, 全连接层
import copy
import os.path
import torchvision
from config import TransformerConfig
import torch.nn.functional as F
from Batch import create_masks
from 杂项 import 打印抽样数据
import pickle
import gc

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N,T, heads, dropout, 最大长度=1024):
        super().__init__()
        self.N = N
        self.T = T
        self.embedX = Embedder(vocab_size, d_model)
        self.embedP = Embedder(最大长度, d_model)
        # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers1 = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.layers2 = get_clones(DecoderLayer(d_model, heads, dropout), T)
        self.norm = Norm(d_model)

    def forward(self, 图向量, 信息汇总, trg_mask):
        position = torch.arange(0, 图向量.size(1), dtype=torch.long,
                                device=图向量.device)

        x = 图向量*0.0+ self.embedP(position) + 信息汇总

        for i in range(self.N):
            x = self.layers1[i](x, trg_mask)
        x =图向量+ self.embedP(position) + x
        for i in range(self.T):
            x = self.layers2[i](x, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, trg_vocab, d_model, N, T,heads, dropout, 图向量尺寸=6 * 6 * 2048):
        super().__init__()
        self.图转A = 全连接层(16*16, d_model)
        self.图转B = 全连接层(d_model, 1)
        self.处理方向 = 全连接层(3, 256)
        #self.处理位置 = 全连接层(4, 256)
        self.处理速度 = 全连接层(1, 256)
        self.综合处理 = 全连接层(d_model, d_model)

        self.decoder = Decoder(trg_vocab, d_model, N,T ,heads, dropout)
        self.动作 = 全连接层(d_model, trg_vocab)
        self.评价 = 全连接层(d_model, 1)


    def forward(self, 状态,device):

        图向量 = self.图转A(torch.from_numpy(状态['图片张量']).cuda(device))
        图向量 = self.图转B(图向量).squeeze(3)
        方向信息 =  F.tanh(self.处理方向(状态['角度集张量_序列']))
        #位置信息 =  F.tanh(self.处理位置(状态['位置张量_序列']))
        速度信息 =  F.tanh(self.处理速度(状态['速度张量_序列']))
        信息汇总 = torch.cat((方向信息,  速度信息), 2)
        信息汇总 = self.综合处理(信息汇总)

        # 图向量 = torch.reshape(图向量, (图向量.shape[0], 图向量.shape[1]))

        d_output = self.decoder(图向量, 信息汇总, 状态['trg_mask'])
        动作 = self.动作(d_output)
        评价 = self.评价(F.tanh(d_output))
        return 动作, 评价


def get_model(opt, trg_vocab, model_weights='model_weights'):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(trg_vocab, opt.d_model, opt.n_layers,opt.t_layers,  opt.heads, opt.dropout)

    if opt.load_weights is not None and os.path.isfile(opt.load_weights + '/' + model_weights):
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/' + model_weights))
    else:
        量 = 0
        for p in model.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p)
                a = 0
            长 = len(p.shape)
            点数 = 1
            for j in range(长):
                点数 = p.shape[j] * 点数

            量 += 点数
        print('使用参数:{}百万'.format(量 / 1000000))
    return model



class PPO_数据集:
    def __init__(self, 并行条目数量):
        self.状态集 = []
        self.动作概率集 = []
        self.评价集 = []
        self.动作集 = []
        self.回报集 = []
        self.完结集 = []

        self.并行条目数量 = 并行条目数量
        self.完整数据={}



    def 提取数据(self):
        状态集_长度 = len(self.状态集)
        条目_起始位 = np.arange(0, 状态集_长度, self.并行条目数量)
        下标集 = np.arange(状态集_长度, dtype=np.int64)
        np.random.shuffle(下标集)  #shuffle 打乱顺序
        条目集 = [下标集[i:i + self.并行条目数量] for i in 条目_起始位]

        return np.array(self.状态集),\
                np.array(self.动作集),\
                np.array(self.动作概率集), \
                self.评价集, \
                np.array(self.回报集),\
                np.array(self.完结集),\
                条目集
    def 记录数据(self, 状态, 动作, 动作概率, 评价, 回报, 完结):
        self.状态集.append(状态)
        self.动作集.append(动作)
        self.动作概率集.append(动作概率)
        self.评价集.append(评价)
        self.回报集.append(回报)
        self.完结集.append(完结)

    def 清除数据(self):
        self.状态集 = []
        self.动作概率集 = []
        self.动作集 = []
        self.回报集 = []
        self.完结集 = []
        self.评价集 = []
        self.完整数据={}
        # del self.状态集,self.动作概率集,self.评价集,self.动作集,self.回报集,self.完结集,self.完整数据
        # gc.collect()

    def 存硬盘(self,文件名):
        self.完整数据['状态集']=self.状态集
        self.完整数据['动作概率集'] = self.动作概率集
        self.完整数据['动作集'] = self.动作集
        self.完整数据['回报集'] = self.回报集
        self.完整数据['完结集'] = self.完结集
        self.完整数据['评价集'] = self.评价集
        save_obj(self.完整数据,文件名)
        self.状态集 = []
        self.动作概率集 = []
        self.动作集 = []
        self.回报集 = []
        self.完结集 = []
        self.评价集 = []
        self.完整数据={}
        # del self.状态集,self.动作概率集,self.评价集,self.动作集,self.回报集,self.完结集,self.完整数据
        # gc.collect()

    def 读硬盘(self,文件名):
        self.完整数据 = load_obj(文件名)
        self.状态集=self.完整数据['状态集']
        self.动作概率集=self.完整数据['动作概率集']
        self.动作集=self.完整数据['动作集']
        self.回报集= self.完整数据['回报集']
        self.完结集= self.完整数据['完结集']
        self.评价集=self.完整数据['评价集']
        self.完整数据={}



def 处理状态参数(状态组,device):

    最长=0
    状态组合={}

   # 操作序列 = np.ones((1,))
    for 状态A in 状态组:
        if 状态A['图片张量'].shape[1]>最长:
            最长=状态A['图片张量'].shape[1]
    for 状态 in 状态组:
        状态A = 状态.copy()
        if 状态A['图片张量'].shape[1] == 最长:
            单元=状态A
            操作序列 = np.ones((最长,))
            遮罩序列 = torch.from_numpy(操作序列.astype(np.int64)).cuda(device).unsqueeze(0)
            单元['遮罩序列']=遮罩序列

        else:
            有效长度=状态A['图片张量'].shape[1]
            差值=最长-有效长度
            形状=状态A['图片张量'].shape
            图片张量_拼接 = torch.zeros(形状[0],差值,形状[2],形状[3]).cuda(device).float()
            图片张量_拼接 = 图片张量_拼接.cpu().numpy()
            状态A['图片张量']=np.append(状态A['图片张量'],图片张量_拼接, axis=1)
            #状态A['图片张量'] = torch.cat((状态A['图片张量'], 图片张量_拼接), 1)
            形状 = 状态A['角度集张量_序列'].shape
            角度集张量_拼接=torch.zeros(形状[0],差值,形状[2]).cuda(device).float()
            状态A['角度集张量_序列'] = torch.cat((状态A['角度集张量_序列'], 角度集张量_拼接), 1)

            形状 = 状态A['位置张量_序列'].shape
            位置张量_拼接=torch.zeros(形状[0],差值,形状[2]).cuda(device).float()
            状态A['位置张量_序列'] = torch.cat((状态A['位置张量_序列'], 位置张量_拼接), 1)

            形状 = 状态A['速度张量_序列'].shape
            速度张量_拼接=torch.zeros(形状[0],差值,形状[2]).cuda(device).float()
            状态A['速度张量_序列'] = torch.cat((状态A['速度张量_序列'], 速度张量_拼接), 1)

            操作序列 = np.ones((有效长度,))
            遮罩序列 = torch.from_numpy(操作序列.astype(np.int64)).cuda(device).unsqueeze(0)
            状态A['遮罩序列']=遮罩序列
            操作序列 = np.ones((差值,))*-1
            遮罩序列 = torch.from_numpy(操作序列.astype(np.int64)).cuda(device).unsqueeze(0)
            状态A['遮罩序列'] = torch.cat((状态A['遮罩序列'], 遮罩序列), 1)
            单元=状态A

        if 状态组合=={}:
            状态组合=单元
        else:
            状态组合['遮罩序列'] = torch.cat((状态组合['遮罩序列'], 单元['遮罩序列']), 0)
            状态组合['速度张量_序列'] = torch.cat((状态组合['速度张量_序列'], 单元['速度张量_序列'],), 0)
            状态组合['位置张量_序列'] = torch.cat((状态组合['位置张量_序列'], 单元['位置张量_序列']), 0)
            状态组合['角度集张量_序列'] = torch.cat((状态组合['角度集张量_序列'], 单元['角度集张量_序列']), 0)
            #状态组合['图片张量'] = torch.cat((状态组合['图片张量'], 单元['图片张量']), 0)
            状态组合['图片张量'] =np.append(状态组合['图片张量'], 单元['图片张量'], axis=0)
    src_mask, trg_mask = create_masks(状态组合['遮罩序列'], 状态组合['遮罩序列'], device)
    状态组合['trg_mask']=trg_mask
    return 状态组合



class 智能体:
    def __init__(self, 动作数, 输入维度, 优势估计参数G=0.99, 学习率=0.0003, 泛化优势估计参数L=0.95,
                 策略裁剪幅度=0.2, 并行条目数=64, 轮数=10,熵系数=0.01):
        self.优势估计参数G = 优势估计参数G
        self.策略裁剪幅度 = 策略裁剪幅度
        self.轮数 = 轮数
        self.熵系数=熵系数
        self.泛化优势估计参数L = 泛化优势估计参数L
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        模型名称 = '模型_动作ppo阶段停bZ1'

        config = TransformerConfig()
        model = get_model(config, 15, 模型名称)
        model = model.cuda(device)
        self.动作 = model

        self.优化函数 = torch.optim.Adam(self.动作.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-9)

        # 模型名称 = '模型_评论'
        #
        # config = TransformerConfig()
        # model = get_model(config, 7, 模型名称)
        # model = model.cuda(device)
        # self.评论 = model
        # self.优化函数_评论 = torch.optim.Adam(self.评论.parameters(), lr=6.25e-5, betas=(0.9, 0.98), eps=1e-9)
        self.数据集 = PPO_数据集(并行条目数)
        self.文件名集=[]

    def 记录数据(self, 状态, 动作, 动作概率, 评价, 回报, 完结):
        self.数据集.记录数据(状态, 动作, 动作概率, 评价, 回报, 完结)
    def 存硬盘(self, 文件名):
        self.数据集.存硬盘(文件名)
        self.文件名集.append(文件名)
    def 读硬盘(self, 文件名):
        self.数据集.读硬盘(文件名)
    def 保存模型(self,轮号):
        print('... 保存模型 ...')

        torch.save(self.动作.state_dict(), 'weights/模型_动作ppo阶段停cZ')
        torch.save(self.动作.state_dict(), 'weights/模型_动作ppo阶段停cZ{}'.format(轮号))
        #torch.save(self.评论.state_dict(), 'weights/模型_评论')

        #torch.save(self.评论.state_dict(), 'weights/模型_评论2')
    def 载入模型(self):
        print('... 载入模型 ...')
        self.动作.载入权重()
        #self.评价.载入权重()

    def 选择动作(self, 状态,device):


        # 分布,q_ = self.动作(状态)
        # r_, 价值 = self.评论(状态)
        self.动作.requires_grad_(False)
        分布, 价值 = self.动作(状态,device)
        价值 = 价值[:, - 1, :]
        分布 = F.softmax(分布, dim=-1)
        分布 = 分布[:, - 1, :]
        分布 = Categorical(分布)
        动作 = 分布.sample()

        动作概率 = T.squeeze(分布.log_prob(动作)).item()
        动作 = T.squeeze(动作).item()
        #价值 = T.squeeze(价值).item()

        return 动作, 动作概率, 价值

        # self.状态集 = []
        # self.probs = []
        # self.动作集 = []
        # self.回报集 = []
        # self.完结集 = []
        # self.评价集 = []
    def 学习(self,device):
        for i in range(1):
        #for 文件名 in self.文件名集:

            #状态集=[]

           # del 状态集
            #gc.collect()
           # self.数据集.清除数据()
           # self.数据集.读硬盘(文件名)


            for _ in range(self.轮数):
                状态集, 动作集, 旧_动作概率集, 评价集, 回报集, 完结集, 条目集 = self.数据集.提取数据()
                print('回报集',回报集[0:10])
                价值 = 评价集

                优势函数值 = np.zeros(len(回报集), dtype=np.float32)

                for t in range(len(回报集) - 1):
                    折扣率 = 1
                    优势值 = 0
                    折扣率 = self.优势估计参数G * self.泛化优势估计参数L
                    计数=0
                    for k in range(t, len(回报集) - 1):

                        优势值 += pow(折扣率, abs(0-计数)) * (回报集[k] + self.优势估计参数G * 价值[k + 1] * (1 - int(完结集[k])) - 价值[k])
                        计数=计数+1
                        if (1 - int(完结集[k]))==0:

                            break
                    优势函数值[t] = 优势值
                    # https://blog.csdn.net/zhkmxx930xperia/article/details/88257891
                    # GAE的形式为多个价值估计的加权平均数
                优势函数值 = T.tensor(优势函数值).to(device)

                价值 = T.tensor(价值).to(device)
                for 条 in 条目集:
                    状态 = 处理状态参数(状态集[条],device)
                    旧_动作概率s = T.tensor(旧_动作概率集[条]).to(device)
                    动作s = T.tensor(动作集[条]).to(device)


                    # 分布, q_ = self.动作(状态)
                    # r_, 评价结果 = self.评论(状态)
                    self.动作.requires_grad_(True)
                    分布, 评价结果 = self.动作(状态,device)
                    分布 = F.softmax(分布, dim=-1)
                    分布 = 分布[:, - 1, :]
                    评价结果 = 评价结果[:, - 1, :]
                    评价结果 = T.squeeze(评价结果)
                    分布 = Categorical(分布)
                    熵损失 = torch.mean(分布.entropy())
                    新_动作概率s = 分布.log_prob(动作s)
                    概率比 = 新_动作概率s.exp() / 旧_动作概率s.exp()
                    # prob_ratio = (new_probs - old_probs).exp()
                    加权概率 = 优势函数值[条] * 概率比
                    加权_裁剪_概率 = T.clamp(概率比, 1 - self.策略裁剪幅度,
                                                     1 + self.策略裁剪幅度) * 优势函数值[条]
                    动作损失 = -T.min(加权概率, 加权_裁剪_概率).mean()

                    总回报 = 优势函数值[条] + 价值[条]
                    评价损失 = (总回报 - 评价结果) ** 2
                    评价损失 = 评价损失 .mean()

                    总损失 = 动作损失 + 0.5 * 评价损失-self.熵系数*熵损失
                    #print(总损失)

                    self.优化函数.zero_grad()
                   # self.优化函数_评论.zero_grad()
                    总损失.backward()
                    self.优化函数.step()
                   # self.优化函数_评论.step()
                print('总损失',总损失)

        self.数据集.清除数据()
        self.文件名集=[]



    def 选择动作_普通程序(self, 速度,轮向角,目标方向_标准化,载具方向_标):
        #print('速度',速度,'轮向角',轮向角,'载具方向_标',载具方向_标,'目标方向_标准化',目标方向_标准化)
        偏差=载具方向_标-目标方向_标准化
        左转 = 0
        右转 = 0
        if 偏差 > 3.14159:
            偏差 = 偏差 - 2 * 3.14159
        elif 偏差 < -3.14159:
            偏差 = 偏差 + 2 * 3.14159
        差角=偏差+轮向角*2
        print('偏差',偏差,'差角',差角,'轮向角',轮向角)

# ---------------------------------------------------------
        if 3.14159/32<差角<3.14159/4 and 轮向角>-0.06:

            if 速度 < 20:
                左转 = 0.5
            elif 速度 > 20:
                左转 = 1

        elif 3.14159/8<差角<3.14159 and 轮向角>-0.06:
            if 速度 < 20:
                左转 = 0.5
            elif 速度 > 20:
                左转 = 1
        elif -3.14159/8<差角<-3.14159/32 and 轮向角<0.06:
            if 速度 < 20:
                右转 = 0.5
            elif 速度 > 20:
                右转 = 1
        elif -3.14159<差角<-3.14159/8 and 轮向角<0.06:
            if 速度 < 20:
                右转 = 0.5
            elif 速度 > 20:
                右转 = 1
 #----------------------------------------

        油门=0
        if 速度<20:
            油门 = 1
        elif 20<速度<30:
            油门 = 0.5
        elif 30 < 速度 < 50:
             油门 = 0
        print('左转',左转,'右转',右转)
        return 油门,左转,右转


    def 监督学习(self, 状态,目标输出,打印,数_词表,操作_分_torch,device):
        分布, 价值 = self.动作(状态,device)
        lin = 分布.view(-1, 分布.size(-1))
        _, 抽样 = torch.topk(分布, k=1, dim=-1)
        抽样np = 抽样.cpu().numpy()

        self.优化函数.zero_grad()
        loss = F.cross_entropy(lin, 目标输出.contiguous().view(-1), ignore_index=-1)
        if 打印:

            print(loss)
            打印抽样数据(数_词表, 抽样np[0:1, :, :], 操作_分_torch[0, :])
        loss.backward()

        self.优化函数.step()


    def 选择动作_old(self, 状态):


        # 分布,q_ = self.动作(状态)
        # r_, 价值 = self.评论(状态)
        输出_实际_A, 价值 = self.动作(状态)


        输出_实际_A = F.softmax(输出_实际_A, dim=-1)
        输出_实际_A = 输出_实际_A[:, - 1, :]
        抽样 = torch.multinomial(输出_实际_A, num_samples=1)
        抽样np = 抽样.cpu().numpy()
        return  抽样np[0,-1]
#item是得到一个元素张量里面的元素值
#优势函数表达在状态s下，某动作a相对于平均而言的优势
#GAE一般优势估计