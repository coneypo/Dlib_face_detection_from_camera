# Delete all the screenshots

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_detection_from_camera

# Created at 2018-10-09

import os

ss = os.listdir("screenshots/")

for image in ss:
    print("Remove: ", "screenshots/"+image)
    os.remove("screenshots/"+image)