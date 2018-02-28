# created: 2018-2-26
# By TimeStamp
# cnblogs: http://www.cnblogs.com/AdaminXie
# GitHub: https://github.com/coneypo/Dlib_face_detection_from_camera

import dlib                     #人脸识别的库dlib
import numpy as np              #数据处理的库numpy
import cv2                      #图像处理的库OpenCv

# dlib预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 创建cv2摄像头对象
cap = cv2.VideoCapture(0)

# cap.set(propId, value)
# 设置视频参数，propId设置的视频参数，value设置的参数值
cap.set(3, 480)

# 截图screenshot的计数器
screenshot_cnt = 0

# cap.isOpened（） 返回true/false 检查初始化是否成功
while(cap.isOpened()):

    # cap.read()
    # 返回两个值：
    #    一个布尔值true/false，用来判断读取视频是否成功/是否到视频末尾
    #    图像对象，一帧图像
    flag, im_rd = cap.read()

    # 每帧数据延时1ms，延时为0读取的是静态帧
    k = cv2.waitKey(1)

    # 取灰度
    img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)

    # 人脸数rects
    rects = detector(img_gray, 0)
    #print(len(rects))

    # 待会要写的字体
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 标68个点
    if(len(rects)!=0):
        # 检测到人脸
        for i in range(len(rects)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(im_rd, rects[i]).parts()])

            for idx, point in enumerate(landmarks):
                # 68点的坐标
                pos = (point[0, 0], point[0, 1])

                # 利用cv2.circle给每个特征点画一个圈，共68个
                cv2.circle(im_rd, pos, 2, color=(0, 255, 0))

                # 利用cv2.putText输出1-68
                cv2.putText(im_rd, str(idx + 1), pos, font, 0.2, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(im_rd, "faces: "+str(len(rects)), (20,50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        # 没有检测到人脸
        cv2.putText(im_rd, "no face", (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    # 添加说明
    im_rd = cv2.putText(im_rd, "s: screenshot", (20, 400), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    im_rd = cv2.putText(im_rd, "q: quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # 按下s键保存
    if (k == ord('s')):
        screenshot_cnt+=1
        cv2.imwrite("screenshot_"+str(screenshot_cnt)+".jpg", im_rd)

    # 按下q键退出
    if(k==ord('q')):
        break

    # 窗口显示
    cv2.imshow("camera", im_rd)

# 释放摄像头
cap.release()

# 删除建立的窗口
cv2.destroyAllWindows()

