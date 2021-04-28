from PIL import Image, ImageQt
import numpy as np
import io
from PyQt5.QtWidgets import QApplication
import sys
import hashlib
def 状态信息综合(图片张量,角度集张量_序列,位置张量_序列,速度张量_序列,trg_mask):
    状态={}
    状态['图片张量']=图片张量[np.newaxis, :, :, :]
    状态['角度集张量_序列']=角度集张量_序列.unsqueeze(0)
    状态['位置张量_序列']=位置张量_序列.unsqueeze(0)
    状态['速度张量_序列']=速度张量_序列.unsqueeze(0)
    状态['trg_mask']=trg_mask
    return 状态

def 解包旧(bin):
    长度=bin[0:4]
    声明长度=int().from_bytes(bin[0:4], byteorder='big', signed=True)
    实际长度=len(bin)
    if 声明长度==实际长度:
        内容=bin[4:实际长度]
        综合信息长度 = int().from_bytes(内容[0:4], byteorder='big', signed=True)
        综合信息=内容[7:综合信息长度].decode()
        图片信息 = 内容[综合信息长度+7: len(内容)]
        #print(hashlib.md5(图片信息).hexdigest())
        image = Image.open(io.BytesIO(图片信息))
        image=image.convert("RGB").resize((640,360),Image.ANTIALIAS)

        图片数组=np.asarray(image)
        #image.show()
        return eval(综合信息),图片数组
def 解包(bin,window):
    长度=bin[0:4]
    声明长度=int().from_bytes(bin[0:4], byteorder='big', signed=True)
    实际长度=len(bin)
    if 声明长度==实际长度:
        内容=bin[4:实际长度]
        综合信息长度 = int().from_bytes(内容[0:4], byteorder='big', signed=True)
        综合信息=内容[7:综合信息长度].decode()
        图片信息 = 内容[综合信息长度+7: len(内容)]
        #print(hashlib.md5(图片信息).hexdigest())
        if len(图片信息 )<2:

            图片数组 = None

        else:
            image = Image.open(io.BytesIO(图片信息))
            image=image.convert("RGB").resize((640,360),Image.ANTIALIAS)


            图片数组=np.asarray(image)
        #image.show()
        return eval(综合信息),图片数组
def 状态信息综合_监督(图片张量,角度集张量_序列,速度张量_序列,trg_mask):
    状态={}
    状态['图片张量']=图片张量
    状态['角度集张量_序列']=角度集张量_序列
    状态['速度张量_序列']=速度张量_序列.unsqueeze(2)
    状态['trg_mask']=trg_mask
    return 状态

def 编号到动作(编号):
    if  编号==0:
        油门=0
        左转=0
        右转=0
    elif 编号==1:
        油门=1
        左转=0
        右转=0
    elif 编号==2:
        油门=0.5
        左转=0
        右转=0
    elif 编号==3:
        油门=0
        左转=1
        右转=0
    elif 编号==4:
        油门=0
        左转=0.5
        右转=0
    elif 编号==5:
        油门=0
        左转=0
        右转=1
    elif 编号==6:
        油门=0
        左转=0
        右转=0.5
    elif 编号==7:
        油门=0.5
        左转=1
        右转=0
    elif 编号==8:
        油门=0.5
        左转=0.5
        右转=0
    elif 编号==9:
        油门=0.5
        左转=0
        右转=1
    elif 编号==10:
        油门=0.5
        左转=0
        右转=0.5
    elif 编号==11:
        油门=1
        左转=1
        右转=0
    elif 编号==12:
        油门=1
        左转=0.5
        右转=0
    elif 编号==13:
        油门=1
        左转=0
        右转=1
    elif 编号==14:
        油门=1
        左转=0
        右转=0.5
    return 油门, 左转, 右转

