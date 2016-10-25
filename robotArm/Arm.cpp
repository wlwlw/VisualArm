#include "Arm.h"

Arm_State newState;
//std_msgs::Float64MultiArray cur_state_msg;
ros::NodeHandle nh;
ros::Subscriber<std_msgs::Float64MultiArray> sub("arm_input_4x3", receive);
//ros::Publisher  arm_state_pub("arm_state_4x3", &cur_state_msg);

float data[12];
bool updated = false;

void receive(const std_msgs::Float64MultiArray& input)
{ 
	digitalWrite(12,HIGH-digitalRead(12));
	//memcpy(&data[0],&input.data[0],sizeof(input.data));
	//cur_state_msg.data = data;
	for(int j=0;j<ARM_INPUT_TENSOR_1;j++){
		newState.vPosition[j]=int(input.data[j]);
		newState.vSpeed[j]=int(input.data[ARM_INPUT_TENSOR_1+j]);
		newState.vIsActive[j]=bool(input.data[2*ARM_INPUT_TENSOR_1+j]);
	}
	updated=true;
}

Arm::Arm()
{
	this->pins[0]=ARM_SERVO_0_PIN;
	this->pins[1]=ARM_SERVO_1_PIN;
	this->pins[2]=ARM_SERVO_2_PIN;
	this->pins[3]=ARM_SERVO_3_PIN;
	for(int i=0;i<ARM_SERVOS_NBR;i++){
		this->state.vPosition[i]=90;
		this->state.vSpeed[i]=1;
		this->state.vIsActive[i]=false;
	}
}

void Arm::init()
{
	pinMode(12,OUTPUT);
	nh.initNode();
	nh.subscribe(sub);
	//nh.advertise(arm_state_pub);
}

void Arm::setState(Arm_State state)
{
	this->state = state;
	for(int i=0;i<ARM_SERVOS_NBR;i++){
		if(this->state.vIsActive[i]==true and this->servos[i].attached() == false) {
			this->servos[i].attach(this->pins[i]);
		}
		else if(this->state.vIsActive[i]==false and this->servos[i].attached() == true) {
			this->servos[i].detach();
		}
		if(this->servos[i].attached()){
			switch(i){
				case 0:
					this->servos[i].write(
						constrain(this->state.vPosition[i], ARM_SERVO_0_LB, ARM_SERVO_0_UB)+ARM_SERVO_0_OFFSET,
						constrain(this->state.vSpeed[i], 1, 255),
						false
					);
					break;
				case 1:
					this->servos[i].write(
						constrain(this->state.vPosition[i], ARM_SERVO_1_LB, ARM_SERVO_1_UB)+ARM_SERVO_1_OFFSET,
						constrain(this->state.vSpeed[i], 1, 255),
						false
					);
					break;
				case 2:
					this->servos[i].write(
						-constrain(this->state.vPosition[i], max(ARM_SERVO_2_LB, state.vPosition[3]-150), min(ARM_SERVO_2_UB,state.vPosition[3]))+ARM_SERVO_2_OFFSET,
						constrain(this->state.vSpeed[i], 1, 255),
						false
					);
					break;
				case 3:
					this->servos[i].write(
						constrain(this->state.vPosition[i], ARM_SERVO_3_LB, ARM_SERVO_3_UB)+ARM_SERVO_3_OFFSET,
						constrain(this->state.vSpeed[i], 1, 255),
						false
					);
					break;
			}
		}
	}
}

void Arm::Update()
{
	nh.spinOnce();
	if(updated){
		this->setState(newState);
		//arm_state_pub.publish(&cur_state_msg);
		updated=false;
	}
}

