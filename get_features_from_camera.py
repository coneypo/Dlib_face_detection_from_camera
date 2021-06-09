# 调用摄像头，进行人脸捕获，和68个特征点的追踪
# Real-time facial landmarks detection from camera

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_detection_from_camera

import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
import time
import os

# 储存截图的目录
path_screenshots = "data/screenshots/"

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
# predictor = dlib.shape_predictor('data/dlib/shape_predictor_5_face_landmarks.dat')

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('data/dlib/shape_predictor_68_face_landmarks.dat')

# cap = cv2.VideoCapture("head-pose-face-detection-male.mp4")   # 输入视频
cap = cv2.VideoCapture("0")                                     # 输入摄像头
cap.set(3, 480)

screenshot_cnt = 0


# Delete all the screenshots
def clear_screenshots():
    ss = os.listdir("data/screenshots/")
    for image in ss:
        print("Remove: ", "data/screenshots/"+image)
        os.remove("data/screenshots/"+image)

# clear_screenshots()


while cap.isOpened():
    flag, im_rd = cap.read()
    k = cv2.waitKey(1)

    img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
    faces = detector(img_gray, 0)

    # 待会要写的字体
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 检测到人脸
    if len(faces) != 0:
        for i in range(len(faces)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(im_rd, faces[i]).parts()])

            # 标 68 个点
            for idx, point in enumerate(landmarks):
                # 68点的坐标
                pos = (point[0, 0], point[0, 1])

                # 特征点画圈
                cv2.circle(im_rd, pos, 2, color=(255, 255, 255))
                # 写1-68
                cv2.putText(im_rd, str(idx + 1), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(im_rd, "Faces: " + str(len(faces)), (20, 50), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        # 没有检测到人脸
        cv2.putText(im_rd, "No face detected", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

    # 添加说明
    im_rd = cv2.putText(im_rd, "'S': screen shot", (20, 400), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    im_rd = cv2.putText(im_rd, "'Q': quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # 按下 's' 键保存
    if k == ord('s'):
        screenshot_cnt += 1
        print(path_screenshots + "ss_" + str(screenshot_cnt) + "_" +
              time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg")
        cv2.imwrite(path_screenshots + "ss_" + str(screenshot_cnt) + "_" +
                    time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg", im_rd)

    # 按下 'q' 键退出
    if k == ord('q'):
        break

    # 窗口显示
    # 参数取 0 可以拖动缩放窗口，为 1 不可以
    # cv2.namedWindow("camera", 0)
    cv2.namedWindow("camera", 1)

    cv2.imshow("camera", im_rd)

# 释放摄像头
cap.release()

# 删除建立的窗口
cv2.destroyAllWindows()