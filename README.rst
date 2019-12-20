Face detection with features points
###################################

Introduction
************

Detect faces from camera, and draw the 5/68 feature points of faces;

#. Use "shape_predictor_5_face_landmarks.dat",
   which is trained on the dlib 5-point face landmark dataset with 7198 faces.
   and identify the corners of the eyes and bottom of the nose:

   .. image:: for_readme/predictor_5_landmarks.jpg
      :align: center


#. Use "shape_predictor_68_face_landmarks.dat", which is trained on the ibug 300-W
   dataset (https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)

   This model is designed to work well with dlib's HOG face detector
   and the CNN face detector.

   It won't work as well when used with a face detector that produces differently aligned boxes

   .. image:: for_readme/predictor_68_landmarks.jpg
      :align: center


* Please install python packages: dlib and numpy at first:

.. code-block:: bash

   pip3 install -r requirements.txt

About Source Code
*****************

Python 源码介绍如下:

#. Use camera in Python / Python OpenCv 调用摄像头;

   .. code-block:: python

      python3 how_to_use_camera.py:

#. Show the 68 features points from local images / 显示本地图像文件中的人脸特征;

   .. code-block:: python

      python3 get_features_from_images.py:


#. Real-time facial landmarks detect and draw feature points /这一步将调用摄像头进行实时人脸检测和特征点绘制;

   .. code-block:: python

      python3 get_features_from_camera.py:


More
****


Author: coneypo

Blog: https://www.cnblogs.com/AdaminXie/p/8472743.html

Mail: coneypo@foxmail.com

Thanks for your support.