#!/usr/bin/env python
#coding: utf-8
import numpy as np
import cv2
import rospy
from std_msgs.msg import Float64MultiArray
from collections import  deque
# 设定红色阈值，HSV空间
redLower = np.array([170, 100, 100])
redUpper = np.array([179, 255, 255])

def getLocationOfRedObject(frame, display=True):
    # 转到HSV空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 根据阈值构建掩膜
    mask = cv2.inRange(hsv, redLower, redUpper)
    # 腐蚀操作
    mask = cv2.erode(mask, None, iterations=2)
    # 膨胀操作，其实先腐蚀再膨胀的效果是开运算，去除噪点
    mask = cv2.dilate(mask, None, iterations=2)
    # 轮廓检测
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # 初始化瓶盖圆形轮廓质心
    center = None
    # 如果存在轮廓
    if len(cnts) > 0:
        # 找到面积最大的轮廓
        c = max(cnts, key=cv2.contourArea)
        # 确定面积最大的轮廓的外接圆
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        # 计算轮廓的矩
        M = cv2.moments(c)
        # 计算质心
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # 只有当半径大于10时，才执行画图
        if radius > 10 and display is True:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    if display: cv2.imshow('Frame', frame)

    if center is not None: return [center[0],center[1],radius]
    else: return None

def main():
    cap = cv2.VideoCapture("/dev/video0")
    if not cap.isOpened():
        cap.open()

    pub1 = rospy.Publisher('camera_input_480x640x3', Float64MultiArray, queue_size=10)
    pub2 = rospy.Publisher('redObject_position_and_size_3x1', Float64MultiArray, queue_size=10)
    rospy.init_node('CameraView', anonymous=True)
    #rospy.init_node('redObject_position_2x1', anonymous=True)
    rate = rospy.Rate(300)  # 30hz
    while not rospy.is_shutdown():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print 'No Camera'
            break
        redObjectInfo = getLocationOfRedObject(frame)
        msg1 = Float64MultiArray(data=frame.flatten())
        if redObjectInfo is not None:
            msg2 = Float64MultiArray(data=redObjectInfo)
            rospy.loginfo(msg2)
            pub2.publish(msg2)

        pub1.publish(msg1)
        # 键盘检测，检测到esc键退出
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        rate.sleep()

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def test():
    cap = cv2.VideoCapture("/dev/video0")
    if not cap.isOpened():
        cap.open()
    ret, frame = cap.read()
    print frame.shape
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
    #test()
