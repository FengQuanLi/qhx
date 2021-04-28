import json
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
def 取随机始终点():
    地图线路 = '地图线路.json'
    with open(地图线路, encoding='utf8') as f:
        总数据={}
        线路 = json.load(f)
        #print(线路)
        未匹配=True
        while 未匹配:
            # 匹配起点
            循环=True
            while 循环:
                随机编号= random.randint(0, 33)
                if 随机编号!=16:
                    循环=False

            起点边号=随机编号
            起点x=线路[str(随机编号)]['起点']['x轴']
            起点y=线路[str(随机编号)]['起点']['y轴']
            终点x=线路[str(随机编号)]['终点']['x轴']
            终点y=线路[str(随机编号)]['终点']['y轴']
            差值x=终点x-起点x
            差值y=终点y-起点y
            任意x= random.randint(30, 70)
            任意y = random.randint(30, 70)
            斜率=0
            if 差值x==0:
                #print('与x轴垂直')
                始发x=起点x
                始发y=起点y+任意y*差值y/100
            elif 差值y==0:
                #print('与x轴平行')
                始发y=起点y
                始发x=起点x+任意x*差值x/100
            else:
                斜率=差值y/差值x
                始发x=起点x + 任意x * 差值x / 100
                始发y = ( 任意x * 差值x / 100 )*斜率+起点y
            #匹配终点
            循环 = True
            while 循环:
                随机编号 = random.randint(0, 33)
                if 随机编号 != 16:
                    循环 = False
            终点边号=随机编号

            起点x=线路[str(随机编号)]['起点']['x轴']
            起点y=线路[str(随机编号)]['起点']['y轴']
            终点x=线路[str(随机编号)]['终点']['x轴']
            终点y=线路[str(随机编号)]['终点']['y轴']
            差值x=终点x-起点x
            差值y=终点y-起点y
            任意x= random.randint(30, 70)
            任意y = random.randint(30, 70)
            斜率=0
            if 差值x==0:
                #print('与x轴垂直')
                终x=起点x
                终y=起点y+任意y*差值y/100
            elif 差值y==0:
                #print('与x轴平行')
                终y=起点y
                终x=起点x+任意x*差值x/100
            else:
                斜率=差值y/差值x
                终x=起点x + 任意x * 差值x / 100
                终y = ( 任意x * 差值x / 100 )*斜率+起点y

            直线距离=((始发x-终x)**2+(始发y-终y)**2)** 0.5
            if 直线距离>300:
                未匹配=False
    节点列=寻路取交汇(始发x, 始发y, 终x, 终y, 起点边号, 终点边号, 线路)
    角度=取初始角度(始发x, 始发y, 终x, 终y, 节点列)
    总数据['始发x']=始发x
    总数据['始发y'] = 始发y
    总数据['终x'] = 终x
    总数据['终y'] = 终y
    总数据['节点列'] = 节点列
    总数据['角度'] = 角度
    return  始发x,始发y,终x,终y,节点列,角度,总数据
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        '/home/fengquanli/.local/share/fonts/锐字真言体.ttf', textSize, encoding="utf-8")
    #"D:/python/辅助/锐字真言体.ttf"
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
def 坐标变换(输入_x,输入_y):
    系数y=358/349.3
    系数x=500/482.9
    中间x=输入_y*系数y
    中间y = 输入_x * 系数x

    偏移量x=225.5*系数x
    偏移量y = 182.2 * 系数y

    输出_y=-int(中间y-偏移量y)
    输出_x = -int(中间x- 偏移量x)
    if 输出_x<0:
        输出_x=0
    if 输出_y<0:
        输出_y=0

    if 输出_x>500:
        输出_x=500
    if 输出_y>358:
        输出_y=358


    return 输出_x, 输出_y
def 寻路取交汇(始发x,始发y,终x,终y,起点边号,终点边号,线路):
    地图交汇点 = '地图交汇点.json'
    交汇单元={}
    当前节点编号=9999
    with open(地图交汇点, encoding='utf8') as f:
        交汇点 = json.load(f)
        #print(交汇点)
    节点列=[]
    弃用点列=[]
    if 起点边号==终点边号:
        return 节点列
    起边_起点编号=线路[str(起点边号)]['起点']['编号']
    起边_终点编号 = 线路[str(起点边号)]['终点']['编号']


    终边_起点编号=线路[str(终点边号)]['起点']['编号']
    终边_终点编号 = 线路[str(终点边号)]['终点']['编号']




    if 起边_起点编号==终边_起点编号 or 起边_起点编号==终边_终点编号:
        print(起边_起点编号)

        交汇单元['x轴']=交汇点[str(起边_起点编号)]['x轴']
        交汇单元['y轴'] = 交汇点[str(起边_起点编号)]['y轴']
        节点列.append(交汇单元.copy())
        return 节点列
    elif 起边_终点编号==终边_起点编号 or 起边_终点编号==终边_终点编号:
        交汇单元['x轴']=交汇点[str(起边_终点编号)]['x轴']
        交汇单元['y轴'] = 交汇点[str(起边_终点编号)]['y轴']
        节点列.append(交汇单元.copy())
        return 节点列
    else:
        #分别计算两组距离之和
        #起点组
        边首点距离到起点距离=((始发x-交汇点[str(起边_起点编号)]['x轴'])**2+(始发y-交汇点[str(起边_起点编号)]['y轴'])**2)**0.5
        边首点距离到终点距离 = ((终x - 交汇点[str(起边_起点编号)]['x轴']) ** 2 + (终y - 交汇点[str(起边_起点编号)]['y轴']) ** 2) ** 0.5
        边首点距离起终点和=边首点距离到起点距离+边首点距离到终点距离

        边尾点距离到起点距离=((始发x-交汇点[str(起边_终点编号)]['x轴'])**2+(始发y-交汇点[str(起边_终点编号)]['y轴'])**2)**0.5
        边尾点距离到终点距离 = ((终x - 交汇点[str(起边_终点编号)]['x轴']) ** 2 + (终y - 交汇点[str(起边_终点编号)]['y轴']) ** 2) ** 0.5
        边尾点距离起终点和=边尾点距离到起点距离+边尾点距离到终点距离
        if 边首点距离起终点和 <= 边尾点距离起终点和:
            交汇单元['x轴'] = 交汇点[str(起边_起点编号)]['x轴']
            交汇单元['y轴'] = 交汇点[str(起边_起点编号)]['y轴']
            节点列.append(交汇单元.copy())
            弃用点列.append(起边_终点编号)
            弃用点列.append(起边_起点编号)
            当前节点编号=起边_起点编号
        else:
            交汇单元['x轴'] = 交汇点[str(起边_终点编号)]['x轴']
            交汇单元['y轴'] = 交汇点[str(起边_终点编号)]['y轴']
            节点列.append(交汇单元.copy())
            弃用点列.append(起边_起点编号)
            弃用点列.append(起边_终点编号)
            当前节点编号 = 起边_终点编号

        最小距离 = 9999999999
        未接邻终点=True
        i=0
        while 未接邻终点:
            i=i+1
            for 相邻单点 in 交汇点[str(当前节点编号)]['相邻点']:

                if 相邻单点['编号'] not in 弃用点列:
                    节点距离到起点距离 = ((相邻单点['x轴'] - 交汇点[str(当前节点编号)]['x轴']) ** 2 + (相邻单点['y轴'] - 交汇点[str(当前节点编号)]['y轴']) ** 2) ** 0.5
                    节点距离到终点距离 = ((终x - 相邻单点['x轴'] ) ** 2 + (终y - 相邻单点['y轴']) ** 2) ** 0.5
                    节点距离起终点和 = 节点距离到起点距离 + 节点距离到终点距离
                    if 最小距离>节点距离起终点和:
                        最小距离=节点距离起终点和
                        最佳相邻点编号=相邻单点['编号']
                        弃用点列.append(最佳相邻点编号)

            if 最佳相邻点编号 == 终边_起点编号 or 最佳相邻点编号 == 终边_终点编号:


                交汇单元['x轴'] = 交汇点[str(最佳相邻点编号)]['x轴']
                交汇单元['y轴'] = 交汇点[str(最佳相邻点编号)]['y轴']
                节点列.append(交汇单元.copy())

                return 节点列
            else:

                交汇单元['x轴'] = 交汇点[str(最佳相邻点编号)]['x轴']
                交汇单元['y轴'] = 交汇点[str(最佳相邻点编号)]['y轴']
                节点列.append(交汇单元.copy())
                当前节点编号=最佳相邻点编号
                弃用点列.append(当前节点编号)
                最小距离 = 9999999999


            if i>23:
                return 节点列

def 取初始角度(起x, 起y, 终x, 终y ,节点列):
    if 节点列==[]:
        差值x = 终x - 起x
        差值y = 终y - 起y
        距离 = ((起x - 终x) ** 2 + (起y - 终y) ** 2) ** 0.5
        #print(差值y/距离)

        角度1=math.acos(差值x / 距离)
        角度2= -角度1
        #print(math.sin(角度1))
        #print(math.sin(角度2))
        if abs(math.sin(角度1)-差值y/距离)<0.001:
            最终角度=角度1
        else:
            最终角度 = 角度2
    else:
        差值x = 节点列[0]['x轴'] - 起x
        差值y = 节点列[0]['y轴']- 起y
        距离 = ((起x - 节点列[0]['x轴']) ** 2 + (起y - 节点列[0]['y轴']) ** 2) ** 0.5
        #print(差值y / 距离)

        角度1 = math.acos(差值x / 距离)
        角度2 = -角度1
        #print(math.sin(角度1))
        #print(math.sin(角度2))
        if abs(math.sin(角度1) - 差值y / 距离) < 0.001:
            最终角度 = 角度1
        else:
            最终角度 = 角度2
    return 最终角度

#{'载具坐标': {'x坐标': -154.63893127441406, 'y坐标': -87.8255615234375}, '碰撞次数': 0, '载具方向角': -3.1397266387939453, '载具速度': 23.10308904270424}
# {'始发x': -154.6603598022461, '始发y': -44.48449661254884, '终x': 53.560214805603025, '终y': -115.02143096923828, '节点列': [{'x轴': -154.63143920898438, 'y轴': -115.02143096923828}, {'x轴': -95.51283264160156, 'y轴': -115.29280853271484}, {'x轴': -25.592313766479492, 'y轴': -115.02143096923828}], '角度': -1.570386320435747}
def 求每步得分(服务器信息,线路总数据,目标距离,目标坐标传入):

   重来=False
   距离得分=0
   偏向角度=0
   载具方向_标准化=0
   目标方向_标准化=0

   if 目标距离==None and 线路总数据['节点列']!=[]:
        目标坐标={}
        距离 = ((服务器信息['载具坐标']['x坐标'] - 线路总数据['节点列'][0]['x轴']) ** 2 + (服务器信息['载具坐标']['y坐标'] - 线路总数据['节点列'][0]['y轴']) ** 2) ** 0.5
        标准方向=取初始角度(服务器信息['载具坐标']['x坐标'],服务器信息['载具坐标']['y坐标'] ,线路总数据['节点列'][0]['x轴'],线路总数据['节点列'][0]['y轴'],[])
        if 距离 >= 30:
            目标坐标['x坐标']=线路总数据['节点列'][0]['x轴']
            目标坐标['y坐标'] = 线路总数据['节点列'][0]['y轴']
            目标坐标['编号'] = 0
        else:
            if len(线路总数据['节点列'])>=2:
                目标坐标['x坐标']=线路总数据['节点列'][1]['x轴']
                目标坐标['y坐标'] = 线路总数据['节点列'][1]['y轴']
                目标坐标['编号'] = 1
                距离 = ((服务器信息['载具坐标']['x坐标'] - 线路总数据['节点列'][1]['x轴']) ** 2 + (
                        服务器信息['载具坐标']['y坐标'] - 线路总数据['节点列'][1]['y轴']) ** 2) ** 0.5
                标准方向 = 取初始角度(服务器信息['载具坐标']['x坐标'], 服务器信息['载具坐标']['y坐标'], 线路总数据['节点列'][1]['x轴'], 线路总数据['节点列'][1]['y轴'],
                             [])
            else:
                目标坐标['x坐标'] = 线路总数据['终x']
                目标坐标['y坐标'] = 线路总数据['终y']
                目标坐标['编号'] = 999999
                距离 = ((服务器信息['载具坐标']['x坐标'] - 线路总数据['终x']) ** 2 + (服务器信息['载具坐标']['y坐标'] - 线路总数据['终y']) ** 2) ** 0.5
                标准方向 = 取初始角度(服务器信息['载具坐标']['x坐标'], 服务器信息['载具坐标']['y坐标'], 线路总数据['终x'], 线路总数据['终y'],
                             [])
   elif 目标距离==None and 线路总数据['节点列']==[]:
        目标坐标={}
        距离 = ((服务器信息['载具坐标']['x坐标'] - 线路总数据['终x']) ** 2 + (服务器信息['载具坐标']['y坐标'] - 线路总数据['终y']) ** 2) ** 0.5
        标准方向 = 取初始角度(服务器信息['载具坐标']['x坐标'], 服务器信息['载具坐标']['y坐标'], 线路总数据['终x'], 线路总数据['终y'],
                     [])
        if 距离>=30:
            目标坐标['x坐标']=线路总数据['终x']
            目标坐标['y坐标'] = 线路总数据['终y']
            目标坐标['编号'] = 999999
            距离得分 = 目标距离 - 距离
        else:
             目标坐标['x坐标'] = 线路总数据['终x']
             目标坐标['y坐标'] = 线路总数据['终y']
             目标坐标['编号'] = 999999
             重来=True
             距离得分 = 目标距离 - 距离


   else:
       if 目标坐标传入['编号']==999999 or 线路总数据['节点列']==[]:
           目标坐标 = {}
           距离 = ((服务器信息['载具坐标']['x坐标'] - 线路总数据['终x']) ** 2 + (服务器信息['载具坐标']['y坐标'] - 线路总数据['终y']) ** 2) ** 0.5
           标准方向 = 取初始角度(服务器信息['载具坐标']['x坐标'], 服务器信息['载具坐标']['y坐标'], 线路总数据['终x'], 线路总数据['终y'],
                        [])
           if 距离 >= 30:
               目标坐标['x坐标'] = 线路总数据['终x']
               目标坐标['y坐标'] = 线路总数据['终y']
               目标坐标['编号'] = 999999
               距离得分 = 目标距离 - 距离
           else:
               目标坐标['x坐标'] = 线路总数据['终x']
               目标坐标['y坐标'] = 线路总数据['终y']
               目标坐标['编号'] = 999999
               重来=True
               距离得分 = 目标距离 - 距离
       else:
           目标坐标 = {}
           a=目标坐标传入['编号']
           距离 = ((服务器信息['载具坐标']['x坐标'] - 线路总数据['节点列'][a]['x轴']) ** 2 + (
                       服务器信息['载具坐标']['y坐标'] - 线路总数据['节点列'][a]['y轴']) ** 2) ** 0.5
           标准方向 = 取初始角度(服务器信息['载具坐标']['x坐标'], 服务器信息['载具坐标']['y坐标'], 线路总数据['节点列'][a]['x轴'],线路总数据['节点列'][a]['y轴'],
                        [])
           if 距离 >= 30:
               目标坐标['x坐标'] = 线路总数据['节点列'][a]['x轴']
               目标坐标['y坐标'] = 线路总数据['节点列'][a]['y轴']
               目标坐标['编号'] = a
               距离得分=目标距离-距离



           else:
               距离得分 = 0
               df=len(线路总数据['节点列'])
               print(df)
               if len(线路总数据['节点列']) > (a+1):
                   目标坐标['x坐标'] = 线路总数据['节点列'][a+1]['x轴']
                   目标坐标['y坐标'] = 线路总数据['节点列'][a+1]['y轴']
                   目标坐标['编号'] = a+1
                   距离 = ((服务器信息['载具坐标']['x坐标'] - 线路总数据['节点列'][a+1]['x轴']) ** 2 + (
                           服务器信息['载具坐标']['y坐标'] - 线路总数据['节点列'][a+1]['y轴']) ** 2) ** 0.5
                   标准方向 = 取初始角度(服务器信息['载具坐标']['x坐标'], 服务器信息['载具坐标']['y坐标'], 线路总数据['节点列'][a+1]['x轴'],
                                线路总数据['节点列'][a+1]['y轴'],
                                [])
               else:
                   目标坐标['x坐标'] = 线路总数据['终x']
                   目标坐标['y坐标'] = 线路总数据['终y']
                   目标坐标['编号'] = 999999
   偏差=服务器信息['载具方向角']-标准方向+3.14159/2
   if 偏差>3.14159:
       偏差=偏差-2*3.14159
   elif 偏差<-3.14159:
       偏差 = 偏差 +2 * 3.14159
   载具方向_标准化=0
   目标方向_标准化=0
   if 服务器信息['载具方向角']>3.14159*2:
       载具方向_标准化= 服务器信息['载具方向角']-3.14159*2
   elif 服务器信息['载具方向角']<0:
       载具方向_标准化 = 服务器信息['载具方向角'] +3.14159 * 2
   else:
       载具方向_标准化 = 服务器信息['载具方向角']

   if 载具方向_标准化>3.14159:
       载具方向_标准化= 载具方向_标准化-3.14159 * 2

   if 标准方向-3.14159/2>3.14159*2:
       目标方向_标准化= 标准方向-3.14159/2-3.14159*2
   elif 标准方向-3.14159/2<0:
       目标方向_标准化 = 标准方向-3.14159/2+3.14159 * 2
   else:
       目标方向_标准化 = 标准方向-3.14159/2

   if 目标方向_标准化>3.14159:
       目标方向_标准化= 目标方向_标准化-3.14159 * 2
   return 重来,距离,目标坐标,距离得分,偏差,载具方向_标准化,目标方向_标准化





def movemouse(event, x, y, flags, param):
    global img
    img2 = img.copy()

    if event == cv2.EVENT_MOUSEMOVE:
        font = cv2.FONT_HERSHEY_SIMPLEX
        message = '{}'.format(img2[y, x])
        img2 = cv2ImgAddText(img2, "起点", 19, 19, (0, 0, 139), 20)
        X, Y,X1,Y1,节点列=取随机始终点()
        print(X, Y)
        X, Y=坐标变换(X, Y)
        X1, Y1 = 坐标变换(X1, Y1)
        print(X, Y)
        cv2.line(img2, (X, Y), (X, Y), (0, 0, 255), 5)
        cv2.line(img2, (X1, Y1), (X1, Y1), (0, 255, 0), 5)
        # cv2.putText(img2, 'car', (19, 19),
        #             font, 0.5, (255, 255, 0), 2)
        cv2.circle(img2, (x, y), 5, (0, 0, 255), -1)
    cv2.imshow('image', img2)


def main():
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", movemouse)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('untitled1.png')
    img_size = img.shape
    h, w = img_size[0:2]
    main()

