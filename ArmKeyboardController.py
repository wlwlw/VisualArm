#!/usr/bin/env python
#coding: utf-8
from evdev import InputDevice
from select import select
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray

key_left,key_right,key_up,key_down = 105, 106,103,108
key_w,key_s,key_a,key_d=17,31,30,32

curState = np.array([[0,0,-40,120],[50,50,50,50],[1,1,1,1]])
graspStep = np.array([[5,0,0,0],[0,0,0,0],[0,0,0,0]])
rotateStep = np.array([[0,5,0,0],[0,0,0,0],[0,0,0,0]])
bigarmStep = np.array([[0,0,0,5],[0,0,0,0],[0,0,0,0]])
smallarmStep = np.array([[0,0,5,0],[0,0,0,0],[0,0,0,0]])


def UpdateStateByInputKey():
    global curState
    dev = InputDevice('/dev/input/event2')
    select([dev], [], [])
    for event in dev.read():
        if event.value == 1 and event.code != 0:
            #print("Key: %s Status: %s" % (event.code, "pressed" if event.value else "release"))
            if event.code==key_left:
                curState[2]=np.array([0,1,0,0])
                curState -= rotateStep
            elif event.code==key_right:
                curState[2]=np.array([0,1,0,0])
                curState += rotateStep
            elif event.code==key_up:
                curState[2]=np.array([0,0,1,0])
                curState += smallarmStep
            elif event.code == key_down:
                curState[2]=np.array([0,0,1,0])
                curState -= smallarmStep
            elif event.code==key_w:
                curState[2]=np.array([0,0,0,1])
                curState += bigarmStep
            elif event.code==key_s:
                curState[2]=np.array([0,0,0,1])
                curState -= bigarmStep
            elif event.code==key_a:
                curState[2]=np.array([1,0,0,0])
                curState += graspStep
            elif event.code==key_d:
                curState[2]=np.array([1,0,0,0])
                curState -= graspStep

def arm_controller():
    pub = rospy.Publisher('arm_input_4x3', Float64MultiArray, queue_size=10)
    rospy.init_node('arm_controller', anonymous=True)
    rate = rospy.Rate(100) # 10hz
    while not rospy.is_shutdown():
        UpdateStateByInputKey()
        #hello_str = "hello world %s" % rospy.get_time()
        #dim = [MultiArrayDimension(label="arm_input",size=12,stride=12)]
        #layout=MultiArrayLayout(dim,0)
        #data = reduce(lambda a,b:a+b, [list(v) for v in curState])
        msg = Float64MultiArray(data = reduce(lambda a,b:a+b, [list(v) for v in curState]))
        rospy.loginfo(msg)
        pub.publish(msg)
        rate.sleep()

def main():
    try:
        arm_controller()
    except rospy.ROSInterruptException:
        pass

from AI.ROSEnvironment import ArmControlPlateform
from config import DQNConfig
from time import sleep
def test():
    config = DQNConfig
    env = ArmControlPlateform(config)
    env.reset()
    for i in range(30):
    	sleep(0.1)
    	action = env.action_space.sample()
        ob,re,ter = env.step(action)
        print(ob,re,ter)

if __name__ == '__main__':
    main()
    
