#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Create two PCA9685 instances with unique I2C addresses
Adafruit_PWMServoDriver pca1 = Adafruit_PWMServoDriver(0x40); // First PCA9685
Adafruit_PWMServoDriver pca2 = Adafruit_PWMServoDriver(0x41); // Second PCA9685

#define SERVOMIN 110
#define SERVOMAX 590

// Calibrated angle → pulse conversion
int angleToPulse(int realAngle) {
  int commandedAngle = map(realAngle, 0, 180, -5, 175);  // your calibration
  return map(commandedAngle, 0, 180, SERVOMIN, SERVOMAX);
}

void setup() {
  Serial.begin(9600);
  Serial.println("=== Two PCA9685 Servo Test (12 Servos) ===");

  // Initialize both PCA9685 boards
  pca1.begin();
  pca2.begin();

  pca1.setPWMFreq(50);  // Standard servo frequency
  pca2.setPWMFreq(50);
  delay(500);

  int angle = 90;
  int pulse = angleToPulse(angle);

  Serial.println("Moving servos to 90° sequentially...");

  // --- PCA9685 #1 : Servos 0–5 ---
  for (int i = 0; i < 11; i++) {
    pca1.setPWM(i, 0, pulse);
    Serial.print("PCA9685 #1 - Servo ");
    Serial.print(i);
    Serial.print(" → 90° | Pulse: ");
    Serial.println(pulse);
    delay(500);
  }

  // --- PCA9685 #2 : Servos 0–5 ---
  for (int i = 0; i < 11; i++) {
    pca2.setPWM(i, 0, pulse);
    Serial.print("PCA9685 #2 - Servo ");
    Serial.print(i);
    Serial.print(" → 90° | Pulse: ");
    Serial.println(pulse);
    delay(500);
  }

  Serial.println("All 12 servos moved to 90°!");
}

void loop() {
  // Nothing in loop
}
