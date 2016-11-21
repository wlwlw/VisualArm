# visual arm

A 4 servo arm control program learn to performreaching task over deep reinforcement Learning. The AI part is based on [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow).

It consists of three ros nodes:

1. CameraView: A python script, use opencv to read image from usb camera(sticked on robot arm), find out the position and size of target object in image, and send them to ArmController node.
2. ArmController: A python script, receiving the information about target object from CameraView, then compute a list of servo angles and send it to Arduino node. 
3. Arduino: arduino program(written in C++), receiving a list of servo angles, and set the servos accrodingly.


## Requirements

- Python 3.3+
- [OpenCV2](http://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Robot Operating System](http://wiki.ros.org/ROS/Installation)
- A multi servo robot arm with Arduino controller.
- An usb camera.
- A red circular gadget used as target object.
- (Recommanded) Ubuntu System.

## Installation

1. Hardware preparation: make sure your robot arm, usb camera are plugged on your computer, all wires are connected properly.
2. Open robotArm/Arm.h, modify the Arm Configrations according to your hardware (PIN, OFFSET...)
3. Open robotArm/robotArm.ino with Arduino Editor, write the program into your arduino board.

## Demo

First, launch roscore:

    $ roscore

Then launch camera:

    $ python CameraView.py --dev=PATH_TO_YOUR_CAMERA_DEV_FILE [--display=True]

Then mount arduino node:

    $ rosrun rosserial_python serial_node.py PATH_TO_YOUR_ARDUINO_USB_DEV_FILE

Finally launch ArmController:

	$ python ArmKeyBoardController.py

or

	$ python ArmRLAIController.py

The former one is a Keyboard Controller, press w/s, a/d, up/down or left/right to change the angles of 4 servos.

The latter one is a Reinforcement AI Controller. When this controller is launched, Robot Arm will begin to move randomly, and hopefully it will gradually learn how to move its head towards the target obeject. Parameters adjustment may be necessary, you can modify them in config.py.

	
