#include <Arduino.h>
#include <ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/MultiArrayLayout.h>
#include <std_msgs/MultiArrayDimension.h>
#include "VarSpeedServo.h"

/****************  Arm Configrations  ****************/
#define ARM_SERVOS_NBR			4
//#define ARM_SERVOS_PIN			0x6c
#define ARM_INPUT_TENSOR_0	3
#define ARM_INPUT_TENSOR_1	4
#define ARM_SERVO_0_PIN     2
#define ARM_SERVO_1_PIN     3
#define ARM_SERVO_2_PIN     5
#define ARM_SERVO_3_PIN     6
#define ARM_SERVO_0_OFFSET  75
#define ARM_SERVO_1_OFFSET  90
#define ARM_SERVO_2_OFFSET  50
#define ARM_SERVO_3_OFFSET  60
#define ARM_SERVO_0_LB      0
#define ARM_SERVO_0_UB      75
#define ARM_SERVO_1_LB      -90
#define ARM_SERVO_1_UB      90
#define ARM_SERVO_2_LB      -90
#define ARM_SERVO_2_UB      45
#define ARM_SERVO_3_LB      0
#define ARM_SERVO_3_UB      120
//#define ARM_STATE_LOWERBOUND  75 0 30 60
//#define ARM_STATE_UPPERBOUND  150 180 150 180			
/*******************************************************/

typedef struct {
	int vPosition[ARM_SERVOS_NBR];
	int vSpeed[ARM_SERVOS_NBR];
	bool vIsActive[ARM_SERVOS_NBR];
} Arm_State;

void receive(const std_msgs::Float64MultiArray& input);

class Arm
{
public:
	int pins[ARM_SERVOS_NBR];
	//ros::Subscriber<std_msgs::Float64MultiArray> sub;
	Arm();
	void init();
	void setState(Arm_State newstate);
	void Update();
	//void receive(const std_msgs::Float64MultiArray& input);
private:
	VarSpeedServo servos[ARM_SERVOS_NBR];
	Arm_State state;
};
