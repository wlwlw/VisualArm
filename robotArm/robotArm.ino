#include "Arm.h"

Arm arm;

//Arm_State a={0,45,-30,30,100,100,100,100,true,true,true,true};
//Arm_State b={0,45,30,30,100,100,100,100,true,true,true,true};
/*Arm_State c={0,-45,-30,10,100,100,100,100,true,true,true,true};
Arm_State d={90,-45,-90,10,100,100,100,100,true,true,true,true};
Arm_State e={90,-45,-30,10,100,100,100,100,true,true,true,true};
Arm_State f={90,45,-30,10,100,100,100,100,true,true,true,true};
*/

void setup() {
  arm.init();
  //arm.setState(test);
  //Serial.println("inited");
}

void loop() {
  arm.Update();
  //arm.setState(a);
  //delay(2000);
  //arm.setState(b);
  //delay(2000);
  /*arm.setState(c);
  delay(2000);
  arm.setState(d);
  delay(2000);
  arm.setState(e);
  delay(2000);
  arm.setState(f);
  delay(2000);
  */
}
