#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// PCA9685 servo drivers
Adafruit_PWMServoDriver pca1 = Adafruit_PWMServoDriver(0x40);
Adafruit_PWMServoDriver pca2 = Adafruit_PWMServoDriver(0x41);

// Configuration
const float SERVO_FREQ = 50.0f;
const int SERVOS_TOTAL = 12;
const int PER_BOARD = 6;

#define SERVOMIN 110
#define SERVOMAX 590

// Persistent angles - NO RESET, NO TIMEOUT
int currentAngles[SERVOS_TOTAL];
char rxbuf[128];
uint8_t rxidx = 0;

// Convert angle (0-180) to PWM pulse
int angleToPulse(int realAngle) {
  // Map with calibration offset
  int commandedAngle = map(realAngle, 0, 180, -5, 175);
  return map(commandedAngle, 0, 180, SERVOMIN, SERVOMAX);
}

// Write to specific servo channel
void writeServoChannel(int servo_index, int pulse) {
  if (servo_index < PER_BOARD) {
    pca1.setPWM(servo_index, 0, pulse);
  } else {
    pca2.setPWM(servo_index - PER_BOARD, 0, pulse);
  }
}

void setup() {
  // Start serial immediately - no waiting
  Serial.begin(115200);
  
  // Initialize I2C and PCA9685 boards
  Wire.begin();
  Wire.setClock(400000); // Fast I2C for responsive control
  
  pca1.begin();
  pca2.begin();
  pca1.setPWMFreq(SERVO_FREQ);
  pca2.setPWMFreq(SERVO_FREQ);
  
  delay(100);

  // Initialize all servos to neutral (90 degrees)
  for (int i = 0; i < SERVOS_TOTAL; ++i) {
    currentAngles[i] = 90;
    writeServoChannel(i, angleToPulse(90));
  }
  
  delay(200);
  
  // Signal ready
  Serial.println("READY");
  Serial.flush();
}

void loop() {
  // Process ALL available bytes immediately - NO QUEUING
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    
    // Ignore carriage return
    if (c == '\r') continue;
    
    // Process complete line
    if (c == '\n') {
      rxbuf[rxidx] = '\0'; // Null terminate
      
      // Parse CSV: expecting exactly 12 comma-separated integers
      int vals[SERVOS_TOTAL];
      int found = 0;
      
      char *ptr = rxbuf;
      char *endptr;
      
      // Fast parsing without tokenization
      while (found < SERVOS_TOTAL && *ptr) {
        // Skip whitespace
        while (*ptr == ' ' || *ptr == ',') ptr++;
        if (!*ptr) break;
        
        // Parse integer
        long val = strtol(ptr, &endptr, 10);
        if (ptr == endptr) break; // No valid number
        
        vals[found++] = (int)val;
        ptr = endptr;
      }
      
      // Apply ONLY if we got exactly 12 values
      if (found == SERVOS_TOTAL) {
        // Validate and apply angles immediately
        bool valid = true;
        for (int i = 0; i < SERVOS_TOTAL; ++i) {
          if (vals[i] < 0 || vals[i] > 180) {
            valid = false;
            break;
          }
        }
        
        if (valid) {
          // Apply to hardware IMMEDIATELY - no buffering
          for (int i = 0; i < SERVOS_TOTAL; ++i) {
            currentAngles[i] = vals[i];
            writeServoChannel(i, angleToPulse(vals[i]));
          }
        }
      }
      
      // Reset buffer for next line
      rxidx = 0;
      
    } else {
      // Add to buffer (with overflow protection)
      if (rxidx < sizeof(rxbuf) - 1) {
        rxbuf[rxidx++] = c;
      } else {
        // Buffer overflow - reset
        rxidx = 0;
      }
    }
  }
  
  // CRITICAL: Minimal delay to prevent blocking
  // Servos maintain last commanded position (PERSISTENT)
  delayMicroseconds(100);
}
