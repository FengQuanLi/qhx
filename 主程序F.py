import socket
import json
import sys
import time, threading
import cv2
import torch
import numpy as np
from cv2显示图片 import 取随机始终点,坐标变换,cv2ImgAddText,求每步得分
from 辅助功能 import 状态信息综合,解包,编号到动作
import torchvision
from resnet_utils import myResnet
from ppo_gpt import 智能体
from Batch import create_masks
import subprocess
from PyQt5.QtWidgets import QApplication
from PIL import Image, ImageQt
import os
window = int(subprocess.check_output(["xdotool", "search" ,"VehiclePhysicsExampleeeveed182"]).decode('ascii').split('\n')[0])


地图交汇点 = '地图交汇点.json'
with open(地图交汇点, encoding='utf8') as f:
    交汇点 = json.load(f)

地图 = cv2.imread("untitled1.png")
接收反馈=True
避免粘包可发=True
图片数组=None
操作辞典={'W按下': 0, 'A按下': 0, 'S按下': 0, 'B按下': 0, 'Z按下': 0, 'D按下': 0, 'J按下': 0, 'K按下': 0, 'L按下': 0, 'M按下': 0, 'P按下': 0, '起x': 0, '起y': 0, '欧拉角Z': -1.57}
服务端打开=True
服务器信息={}
发送开关=True
更新起点_目标=True
碰撞扣分=False
图片数组应急=[]
应急次数=0
心跳=True
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
mod = torchvision.models.resnet101(pretrained=True).eval().cuda(device).requires_grad_(False)
resnet101 = myResnet(mod)
N = 15000 # 运行N次后学习
条数 = 32
轮数 = 3
学习率 = 0.0003
智能体 = 智能体(动作数=7, 并行条目数=条数,
          学习率=学习率, 轮数=轮数,
          输入维度=6)
接收反馈=True
总次数=0
阶段分=0
阶段分2=0
def 启动TCP客户端():
    bin=None
    global 操作辞典, 服务端打开,服务器信息,发送开关,接收反馈,避免粘包可发,图片数组
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', 6666))
    while True:
        if 发送开关 and 避免粘包可发:
            try:

                data = str(操作辞典)
                编码结果=data.encode()
                s.send(编码结果)
                避免粘包可发=False
                发送开关 = False
                操作辞典['J按下'] = 0
                操作辞典['K按下'] = 0
                操作辞典['L按下'] = 0
                操作辞典['M按下'] = 0
                操作辞典['P按下'] = 0
                操作辞典['D按下'] = 0
                操作辞典['A按下'] = 0
                操作辞典['B按下'] = 0
                操作辞典['Z按下'] = 0
            except:
                服务端打开 = False

                print("Dialogue Over")
                s.close()
                sys.exit(0)
        if 接收反馈:
            开始1=time.time()
            try:
                while True:

                    buf = s.recv(1024)
                    if bin==None:
                        bin=buf

                        声明长度 = int().from_bytes(bin[0:4], byteorder='big', signed=True)
                        if 声明长度==len(bin):
                            break

                        else:
                            pass
                    else:
                        声明长度 = int().from_bytes(bin[0:4], byteorder='big', signed=True)
                        if 声明长度 == len(bin+buf):
                            bin = bin + buf
                            break
                        elif 声明长度 > len(bin+buf):

                            bin = bin + buf
                            #print(len(bin))
                        else:
                            break
                try:
                    服务器信息,图片数组=解包(bin,window)
                    图片数组应急=图片数组
                except:
                    应急次数=应急次数+1
                    print('应急次数',应急次数)
                    图片数组= 图片数组应急
                bin = None
                接收反馈 = False
                避免粘包可发 = True
            except:
                服务端打开 = False
                print("Dialogue Over")
                s.close()
                sys.exit(0)
        else:
            if 心跳!=True:
                break
            time.sleep(0.01)
            pass

def CV信息显示():
    global 地图,服务器信息,更新起点_目标,操作辞典,线路总数据
    imgNew = 地图.copy()


    for i in range(10000000):

        if 更新起点_目标:
            起x, 起y, 终x, 终y ,节点列,角度,线路总数据= 取随机始终点()

            操作辞典['起x']=起x
            操作辞典['起y'] = 起y
            操作辞典['欧拉角Z'] = 角度-3.14159/2
            #print("角度",角度)
            起x, 起y = 坐标变换(起x, 起y)
            终x, 终y = 坐标变换(终x, 终y)
            imgNew1 = cv2ImgAddText(imgNew, "起", 起x, 起y, (111, 255, 0), 15)
            imgNew1 = cv2ImgAddText(imgNew1, "终", 终x, 终y, (0, 111, 255), 15)
            p = 0
            #print('节点列',节点列)
            for 节点 in 节点列:
                X,Y=坐标变换(节点['x轴'], 节点['y轴'])

                imgNew1 = cv2ImgAddText(imgNew1, str(p), X, Y, (155, 111, 0), 15)
                p=p+1

            更新起点_目标 = False
            操作辞典['P按下'] = 1
        if 服务器信息 !={}:

            X,Y=坐标变换(服务器信息['载具坐标']['x坐标'],服务器信息['载具坐标']['y坐标'])

            imgNew2=imgNew1.copy()

            cv2.line(imgNew2, (X, Y), (X, Y), (0, 0, 255), 5)
            cv2.imshow('aaa', imgNew2)
            cv2.waitKey(1)


        time.sleep(0.1)


TCP客户端= threading.Thread(target=启动TCP客户端)
TCP客户端.start()
CV信息= threading.Thread(target=CV信息显示)
CV信息.start()
步数=0
学习次数=0
分数记录 = []
速度记录=[]
最高分=0
time.sleep(1)
app = QApplication(sys.argv)
screen = app.primaryScreen()
动作_状态辞典={"图片号":0,"油门":0,"左转":0,"右转":0,"载具速度":0,"轮向角":0,"载具方向":0,"目标方向":0}
训练数据保存目录='../训练数据2'

for i in range(6666666):

    图片路径 = 训练数据保存目录 + '/{}/'.format(str(int(time.time())))
    os.mkdir(图片路径)
    记录文件 = open(图片路径 + '_操作数据.json', 'w+')

    操作辞典 = {'W按下': 0, 'A按下': 0, 'B按下': 0, 'Z按下': 0, 'S按下': 0, 'D按下': 0, 'J按下': 0, 'K按下': 0, 'L按下': 0, 'M按下': 0, 'P按下': 0, '起x': 0, '起y': 0,
            '欧拉角Z': -1.57}
    更新起点_目标=True
    time.sleep(0.1)
    避免粘包可发 = True
    发送开关 = True
    time.sleep(0.1)
    接收反馈=True
    角度集张量 = torch.FloatTensor(1, 3).cuda(device)
    位置张量 = torch.FloatTensor(1, 4).cuda(device)
    速度张量 = torch.FloatTensor(1, 1).cuda(device)
    图片张量 = torch.Tensor(0)
    操作序列 = np.ones((1,))
    目标距离 = None
    目标坐标传入 = {}
    角度集张量_序列 = None
    位置张量_序列 = None
    速度张量_序列 = None
    单轮计数 = 0
    单轮计数2= 0
    排除连续=0
    分数=0
    计数2 = 0
    低速度=False
    time.sleep(0.1)
    重来, 目标距离, 目标坐标传入, 距离得分, 偏差, 载具方向_标准化, 目标方向_标准化 = 求每步得分(服务器信息, 线路总数据, 目标距离, 目标坐标传入)

    角度集张量[0, 2] = 目标方向_标准化
    角度集张量[0, 1] = 载具方向_标准化
    角度集张量[0, 0] = 服务器信息['轮向角']
    位置张量[0, 0] = 目标坐标传入['x坐标']
    位置张量[0, 1] = 目标坐标传入['y坐标']
    位置张量[0, 2] = 服务器信息['载具坐标']['x坐标']
    位置张量[0, 3] = 服务器信息['载具坐标']['y坐标']
    速度张量[0, 0] = 服务器信息['载具速度']


    img = screen.grabWindow(window)
    image = ImageQt.fromqimage(img)
    image=image.resize((640, 360))
    #image.save("采样.jpg")
    图片数组=np.asarray(image)
    截屏 = torch.from_numpy(图片数组).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
    _, out = resnet101(截屏)
    out = torch.reshape(out, (1, 512, 16*16))
    图片张量 = out
    角度集张量_序列 = 角度集张量
    位置张量_序列 = 位置张量
    速度张量_序列 = 速度张量
    操作张量 = torch.from_numpy(操作序列.astype(np.int64)).cuda(device)
    src_mask, trg_mask = create_masks(操作张量.unsqueeze(0), 操作张量.unsqueeze(0), device)
    状态 = 状态信息综合(图片张量.cpu().numpy(), 角度集张量_序列, 位置张量_序列, 速度张量_序列, trg_mask)
    完结=False
    局内计数=0
    while not 完结:
        计时开始 = time.time()
        步数 += 1
        if 步数 % N == 0:
            操作辞典['B按下'] = 1

        动作, 动作可能性, 评价 = 智能体.选择动作(状态,device)

        油门, 左转, 右转 = 编号到动作(动作)


        操作辞典['S按下'] = 0
        操作辞典['W按下'] = 油门
        操作辞典['A按下'] = 左转
        操作辞典['D按下'] = 右转
        #print('油门:{}  左转:{} 右转:{} 速度：{}km/h'.format(油门,左转,右转,int(服务器信息['载具速度'])))
        发送开关 = True
        避免粘包可发 = True
        time.sleep(0.02)
        接收反馈 = True
        time.sleep(0.02)

        if (服务器信息['碰撞次数'] != 0 or abs(服务器信息['载具方向角x']) > 0.5 or abs(服务器信息['载具方向角y']) > 0.2) and 局内计数>1:

            排除连续 = 总次数
            碰撞扣分 = True
            完结=True
            print('重来--。。。。。。。。。。。。。。。。碰撞')

        总次数 = 总次数 + 1

        重来, 目标距离, 目标坐标传入, 距离分, 偏差, 载具方向_标准化, 目标方向_标准化 = 求每步得分(服务器信息, 线路总数据, 目标距离, 目标坐标传入)
        if abs(距离分)>9:
            距离分=0

        阶段分 = 阶段分 + 服务器信息['载具速度']
        if 单轮计数 == 20:
            单轮计数 = 0


            if 阶段分 < 20:
                低速度=True

                完结 = True
                print('重来-。。。。。。。。。。。。。。。。。。。。。低速','动作',动作)
            阶段分 = 0

        阶段分2 = 阶段分2 + 距离分
        if 单轮计数2 == 20:
            单轮计数2 = 0


            if 阶段分2 < 6:
                低速度=True

                完结 = True
                print('重来-。。。。。。。。。。。。。。。。。。。。。低位移','动作',动作)
            阶段分2 = 0
        if 3>距离分 > 1:
            距离得分 =距离分
        if 距离分 > 3:
            距离得分 =距离分
        if -1<距离分 <-0.2:
            距离得分 =距离分
            #距离得分 = 0
        if 距离分 <-1:
            距离得分 =距离分
            #距离得分 = 0
        if 0.2<距离分<1:
            距离得分=距离分
        if 0<距离分<0.2:
            距离得分=距离得分
        if -0.2< 距离分 < 0:
            距离得分 = 距离分
            #距离得分 = 0
        if 碰撞扣分:
            距离得分=-20
            碰撞扣分=False
        if 重来:
            完结 = True
            距离得分 = 20
            print('重来---。。。。。。。。。。。。***********。胜利')
        if 低速度:
            距离得分 = -0
            低速度=False
        #print('距离得分', 距离得分)
        单轮计数 += 1
        单轮计数2 += 1
        角度集张量[0, 2] = 目标方向_标准化
        角度集张量[0, 1] = 载具方向_标准化
        角度集张量[0, 0] = 服务器信息['轮向角']
        位置张量[0, 0] = 目标坐标传入['x坐标']
        位置张量[0, 1] = 目标坐标传入['y坐标']
        位置张量[0, 2] = 服务器信息['载具坐标']['x坐标']
        位置张量[0, 3] = 服务器信息['载具坐标']['y坐标']
        速度张量[0, 0] = 服务器信息['载具速度']


        img = screen.grabWindow(window)
        image = ImageQt.fromqimage(img)
        image = image.resize((640, 360))
        #image.save("采样.jpg")

        图片数组 = np.asarray(image)
        截屏 = torch.from_numpy(图片数组).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
        _, out = resnet101(截屏)
        out = torch.reshape(out, (1, 512, 16*16))
        计数2 = 计数2 + 1
        if 图片张量.shape[0] == 0:

            图片张量 = out
            角度集张量_序列 = 角度集张量
            位置张量_序列 = 位置张量
            速度张量_序列 = 速度张量


        elif 图片张量.shape[0] < 1:

            图片张量 = torch.cat((图片张量, out), 0)
            角度集张量_序列 = torch.cat((角度集张量_序列, 角度集张量), 0)
            位置张量_序列 = torch.cat((位置张量_序列, 位置张量), 0)
            速度张量_序列 = torch.cat((速度张量_序列, 速度张量), 0)
            操作序列 = np.append(操作序列, 1)


        else:

            图片张量 = 图片张量[0:0, :]
            操作序列 = 操作序列[0:0]
            角度集张量_序列 = 角度集张量_序列[0:0]
            位置张量_序列 = 位置张量_序列[0:0]
            速度张量_序列 = 速度张量_序列[0:0]
            操作序列 = np.append(操作序列, 1)
            图片张量 = torch.cat((图片张量, out), 0)
            角度集张量_序列 = torch.cat((角度集张量_序列, 角度集张量), 0)
            位置张量_序列 = torch.cat((位置张量_序列, 位置张量), 0)
            速度张量_序列 = torch.cat((速度张量_序列, 速度张量), 0)
        # out= torch.reshape(out, (768, 96))
        # print('得分', 得分, '更新起点_目标', 更新起点_目标,type(截屏))
        操作张量 = torch.from_numpy(操作序列.astype(np.int64)).cuda(device)
        src_mask, trg_mask = create_masks(操作张量.unsqueeze(0), 操作张量.unsqueeze(0), device)
        状态_ = 状态信息综合(图片张量.cpu().numpy(), 角度集张量_序列, 位置张量_序列, 速度张量_序列, trg_mask)
        分数 += 距离得分
        智能体.记录数据(状态, 动作, 动作可能性, 评价, 距离得分, 完结)
        速度记录.append(服务器信息['载具速度'])
        if 步数 % N == 0:
            心跳 = False
            # 伪心跳_ = threading.Thread(target=伪心跳)
            # 伪心跳_.start()
            print('保存中。。。。。。。。。。。。。。。。')
            #智能体.存硬盘(str(步数))
            print('保存完毕。。。。。。。。。。。。。。。。')

            if 步数 % (N*1) == 0:
                print('学习中。。。。。。。。。。。。。。。。')
                智能体.学习(device)
                print('分数', 分数)
                智能体.保存模型(学习次数)
                分数记录 = []
                速度记录=[]
                print('学习完毕。。。。。。。。。。。。。。。。')


                #智能体.保存模型(学习次数)
            学习次数 += 1
            心跳=True
            TCP客户端 = threading.Thread(target=启动TCP客户端)
            TCP客户端.start()

            time.sleep(0.2)


            操作辞典['Z按下'] = 1





        状态=状态_
        延迟 = 0.22 - (time.time() - 计时开始)
        if 延迟 > 0:
            time.sleep(延迟)
        局内计数 = 局内计数 + 1

    分数记录.append(分数)

    平均分 = np.mean(分数记录[-500:])
    平均速度 = np.mean(速度记录[-15000:])
    if 平均分 > 最高分:
        最高分 = 平均分

    print('步数', 步数, '平均分', 平均分,'最高分',最高分,'局数',i,'平均速度',平均速度)










    #time.sleep(2)
    # while True:
    #
    #     time.sleep(11)




