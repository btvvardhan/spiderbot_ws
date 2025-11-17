#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pca1 = Adafruit_PWMServoDriver(0x40);
Adafruit_PWMServoDriver pca2 = Adafruit_PWMServoDriver(0x41);

#define SERVOMIN 110
#define SERVOMAX 590

int angleToPulse(int realAngle) {
  int commandedAngle = map(realAngle, 0, 180, -5, 175);
  return map(commandedAngle, 0, 180, SERVOMIN, SERVOMAX);
}

void setup() {
  // MATCH YOUR PYTHON: 115200 baud
  Serial.begin(115200);
  Serial.println("BOOT");
  
  Wire.begin();
  pca1.begin();
  pca2.begin();
  pca1.setPWMFreq(50);
  pca2.setPWMFreq(50);
  delay(100);
  
  // Test: Move all to 90°
  Serial.println("TEST: Moving to 90");
  for (int i = 0; i < 12; i++) {
    int pulse = angleToPulse(90);
    if (i < 6) {
      pca1.setPWM(i, 0, pulse);
    } else {
      pca2.setPWM(i - 6, 0, pulse);
    }
  }
  delay(1000);
  
  // Test: Move all to 120°
  Serial.println("TEST: Moving to 120");
  for (int i = 0; i < 12; i++) {
    int pulse = angleToPulse(120);
    if (i < 6) {
      pca1.setPWM(i, 0, pulse);
    } else {
      pca2.setPWM(i - 6, 0, pulse);
    }
  }
  delay(1000);
  
  // Back to 90°
  Serial.println("TEST: Back to 90");
  for (int i = 0; i < 12; i++) {
    int pulse = angleToPulse(90);
    if (i < 6) {
      pca1.setPWM(i, 0, pulse);
    } else {
      pca2.setPWM(i - 6, 0, pulse);
    }
  }
  
  Serial.println("READY for serial commands");
}

char rxbuf[256];
int rxidx = 0;

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    
    // Echo what we receive (so we can see if data is arriving)
    Serial.print("RX:");
    Serial.println((int)c);
    
    if (c == '\r') continue;
    
    if (c == '\n') {
      rxbuf[rxidx] = 0;
      Serial.print("LINE:[");
      Serial.print(rxbuf);
      Serial.println("]");
      
      // Parse
      int vals[12];
      int found = 0;
      char temp[256];
      strcpy(temp, rxbuf);
      char *tok = strtok(temp, ",");
      
      while (tok && found < 12) {
        vals[found++] = atoi(tok);
        tok = strtok(NULL, ",");
      }
      
      Serial.print("PARSED:");
      Serial.println(found);
      
      if (found == 12) {
        Serial.println("MOVING!");
        for (int i = 0; i < 12; i++) {
          int pulse = angleToPulse(vals[i]);
          if (i < 6) {
            pca1.setPWM(i, 0, pulse);
          } else {
            pca2.setPWM(i - 6, 0, pulse);
          }
        }
        Serial.println("DONE");
      } else {
        Serial.println("ERROR: Wrong count");
      }
      
      rxidx = 0;
    } else {
      if (rxidx < 255) {
        rxbuf[rxidx++] = c;
      }
    }
  }
  
  delay(1);
}
