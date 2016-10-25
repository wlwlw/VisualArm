__author__ = 'Bright'
import numpy as np
import thread
import rospy
from std_msgs.msg import Float64MultiArray
from random import randrange
desired_target_postion = np.array([0.5,0.5,0.8]) #in the middle of screen with its diameter equal to about 30% of screen
ep = 0.1 #done when square distance to the desired postion smaller than ep

# avaliable action step for changing arm state
graspStep = np.array([5,0,0,0])
rotateStep = np.array([0,5,0,0])
bigarmStep = np.array([0,0,0,5])
smallarmStep = np.array([0,0,5,0])
# default arm movement speed
speed = 255

class box(object):
    _high,_low,high,low=None,None,None,None
    def __init__(self,iterable):
        self._dim = np.array(iterable)
    def __iter__(self):
        return self._dim
    def __getitem__(self, item):
        return self._dim[item]

class discrete(object):
    def __init__(self,number):
        self.n=number
    def sample(self):
        return randrange(self.n)

class ArmControlPlateform(object):
    def __init__(self, config):
        self.name = config.env_name
        self.display = config.display
        self.observation_space = box([7]) #angluar position of 4 servos, x-y postion and size of target on screen
        # observation_space before normlization
        self.observation_space._high = np.array([75,90,45,120,config.camera_res_y,config.camera_res_x,config.camera_res_y])
        self.observation_space._low = np.array([0,-90,-90,0,0,0,0])
        # after normalization
        self.observation_space.high = np.array([1.]*self.observation_space[0])
        self.observation_space.low = np.array([0.]*self.observation_space[0])

        self.action_space = discrete(8) #add or substract op of the angular of 4 servos.

        self.controlTopic = rospy.Publisher(config.control_topic, Float64MultiArray, queue_size=10)
        rospy.Subscriber(config.control_topic, Float64MultiArray, self._get_arm_state)
        rospy.Subscriber(config.camera_topic, Float64MultiArray, self._get_target_info)
        
        self.resetMsg = Float64MultiArray(data=[0.,0.,-30.,90.,200,200,200,200,1,1,1,1])
        self._observation = np.array([0.5]*self.observation_space[0]) #original observation
        self._oldStateScore = self._stateScore

        self._actionMsgDict = [
            graspStep,rotateStep,bigarmStep,smallarmStep,
            -graspStep,-rotateStep,-bigarmStep,-smallarmStep
        ]

        rospy.init_node(self.name, anonymous=True)
        self.rate = rospy.Rate(25)
        thread.start_new_thread(rospy.spin, ())


    def reset(self, from_random_game=False):
        self.controlTopic.publish(self.resetMsg)
        #print("reseted")
        return self.observation

    def step(self, action):
        self.controlTopic.publish(self._generateActionMsg(action))
        self.rate.sleep()
        return self.observation, self.reward, self.done

    def normalize(self, _observation):
        #print(_observation,self.observation_space._low)
        return(np.maximum.reduce([
            np.array([0.]*self.observation_space[0]),
            np.minimum.reduce([
                np.array([1.]*self.observation_space[0]),
                (_observation-self.observation_space._low)/(self.observation_space._high-self.observation_space._low)
                ])
            ])
        )

    @property
    def observation(self):
        return self.normalize(self._observation)

    @property
    def _armState(self):
        return self._observation[:4]

    @property
    def reward(self):
        reward = self._stateScore-self._oldStateScore
        self.oldStateScore = self._stateScore
        return reward

    @property
    def done(self):
        return 0.-self._stateScore < ep

    @property
    def _stateScore(self):
        return -np.sum(np.square(self.observation[4:]-desired_target_postion))

    def _get_arm_state(self, data):
        #print("get_arm_state_is_called")
        self._observation[:4]=np.array(data.data[:4])

    def _get_target_info(self,data):
        #print("get_target_info is called")
        self._observation[4:]=np.array(data.data)

    def _generateActionMsg(self, action):
        targetState = self._armState+self._actionMsgDict[action]
        return Float64MultiArray(data=list(targetState)+[speed]*4+[1]*4)


