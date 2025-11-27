#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ====== PCA9685 SETUP (SERVOS) ======
Adafruit_PWMServoDriver pca1 = Adafruit_PWMServoDriver(0x40);
Adafruit_PWMServoDriver pca2 = Adafruit_PWMServoDriver(0x41);

#define SERVOMIN 110
#define SERVOMAX 590

const float SERVO_FREQ = 50.0f;
const int SERVOS_TOTAL = 12;
const int PER_BOARD = 6;

// ====== MPU9250 SETUP (IMU) ======
#define MPU9250_ADDR  0x68  // Default address (or 0x69 if AD0 is high)
#define AK8963_ADDR   0x0C

// MPU9250 registers
#define MPU9250_PWR_MGMT_1  0x6B
#define MPU9250_CONFIG      0x1A
#define MPU9250_GYRO_CONFIG 0x1B
#define MPU9250_ACCEL_CONFIG 0x1C
#define MPU9250_INT_PIN_CFG 0x37
#define MPU9250_WHO_AM_I    0x75

// AK8963 registers
#define AK8963_CNTL1        0x0A

// ====== IMU STATE MACHINE ======
enum IMUState {
  IMU_IDLE,
  IMU_READ_ACCEL,
  IMU_READ_GYRO,
  IMU_READ_MAG,
  IMU_PUBLISH
};

IMUState imuState = IMU_IDLE;
unsigned long lastIMUTime = 0;
const int IMU_INTERVAL = 50; // 50ms = 20Hz
bool imuAvailable = false;

// IMU data buffers
int16_t ax, ay, az, gx, gy, gz, mx, my, mz;

// ====== SERVO STATE ======
int currentAngles[SERVOS_TOTAL];
char rxbuf[128];
uint8_t rxidx = 0;

// ====== I2C HELPERS ======
inline void writeByte(uint8_t addr, uint8_t reg, uint8_t data) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(data);
  Wire.endTransmission();
}

inline uint8_t readByte(uint8_t addr, uint8_t reg) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(addr, (uint8_t)1);
  return Wire.read();
}

inline void readBytes(uint8_t addr, uint8_t reg, uint8_t count, uint8_t *dest) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(addr, count);
  uint8_t i = 0;
  while (Wire.available() && i < count) {
    dest[i++] = Wire.read();
  }
}

// ====== IMU INIT ======
bool initMPU9250() {
  // Check WHO_AM_I
  uint8_t whoami = readByte(MPU9250_ADDR, MPU9250_WHO_AM_I);
  if (whoami != 0x71 && whoami != 0x73) {
    return false; // MPU9250 not found
  }
  
  // Wake up
  writeByte(MPU9250_ADDR, MPU9250_PWR_MGMT_1, 0x00);
  delay(100);
  
  // Configure
  writeByte(MPU9250_ADDR, MPU9250_CONFIG, 0x03);
  writeByte(MPU9250_ADDR, MPU9250_GYRO_CONFIG, 0x00);    // ±250 dps
  writeByte(MPU9250_ADDR, MPU9250_ACCEL_CONFIG, 0x00);   // ±2g
  writeByte(MPU9250_ADDR, MPU9250_INT_PIN_CFG, 0x02);    // Enable magnetometer
  
  // Init magnetometer
  writeByte(AK8963_ADDR, AK8963_CNTL1, 0x00);
  delay(10);
  writeByte(AK8963_ADDR, AK8963_CNTL1, 0x16); // 16-bit, 100Hz
  delay(10);
  
  return true;
}

// ====== NON-BLOCKING IMU READ (STATE MACHINE) ======
void updateIMU() {
  if (!imuAvailable) return;
  
  unsigned long now = millis();
  
  switch (imuState) {
    case IMU_IDLE:
      // Check if it's time to start new reading
      if (now - lastIMUTime >= IMU_INTERVAL) {
        imuState = IMU_READ_ACCEL;
      }
      break;
    
    case IMU_READ_ACCEL: {
      // Quick read: accel (6 bytes, ~1ms)
      uint8_t rawData[6];
      readBytes(MPU9250_ADDR, 0x3B, 6, rawData);
      ax = ((int16_t)rawData[0] << 8) | rawData[1];
      ay = ((int16_t)rawData[2] << 8) | rawData[3];
      az = ((int16_t)rawData[4] << 8) | rawData[5];
      imuState = IMU_READ_GYRO;
      break;
    }
    
    case IMU_READ_GYRO: {
      // Quick read: gyro (6 bytes, ~1ms)
      uint8_t rawData[6];
      readBytes(MPU9250_ADDR, 0x43, 6, rawData);
      gx = ((int16_t)rawData[0] << 8) | rawData[1];
      gy = ((int16_t)rawData[2] << 8) | rawData[3];
      gz = ((int16_t)rawData[4] << 8) | rawData[5];
      imuState = IMU_READ_MAG;
      break;
    }
    
    case IMU_READ_MAG: {
      // Quick read: magnetometer (6 bytes, ~1ms)
      uint8_t rawData[6];
      readBytes(AK8963_ADDR, 0x03, 6, rawData);
      mx = ((int16_t)rawData[1] << 8) | rawData[0];
      my = ((int16_t)rawData[3] << 8) | rawData[2];
      mz = ((int16_t)rawData[5] << 8) | rawData[4];
      imuState = IMU_PUBLISH;
      break;
    }
    
    case IMU_PUBLISH: {
      // Convert and publish
      float ax_ms2 = (ax / 16384.0) * 9.81;
      float ay_ms2 = (ay / 16384.0) * 9.81;
      float az_ms2 = (az / 16384.0) * 9.81;
      
      float gx_rads = (gx / 131.0) * (PI / 180.0);
      float gy_rads = (gy / 131.0) * (PI / 180.0);
      float gz_rads = (gz / 131.0) * (PI / 180.0);
      
      // Send via Serial
      Serial.print("IMU,");
      Serial.print(ax_ms2, 4); Serial.print(",");
      Serial.print(ay_ms2, 4); Serial.print(",");
      Serial.print(az_ms2, 4); Serial.print(",");
      Serial.print(gx_rads, 4); Serial.print(",");
      Serial.print(gy_rads, 4); Serial.print(",");
      Serial.print(gz_rads, 4); Serial.print(",");
      Serial.print(mx); Serial.print(",");
      Serial.print(my); Serial.print(",");
      Serial.print(mz);
      Serial.println();
      
      // Reset state
      lastIMUTime = now;
      imuState = IMU_IDLE;
      break;
    }
  }
}

// ====== SERVO FUNCTIONS (UNCHANGED) ======
int angleToPulse(int realAngle, int servo_index) {
  if (servo_index == 2 || servo_index == 5 || servo_index == 8 || servo_index == 11) {
    realAngle = 180 - realAngle;
  }
  int commandedAngle = map(realAngle, 0, 180, -5, 175);
  return map(commandedAngle, 0, 180, SERVOMIN, SERVOMAX);
}

void writeServoChannel(int servo_index, int angle) {
  if (servo_index > 11) return;
  int pulse = angleToPulse(angle, servo_index);
  
  if (servo_index < PER_BOARD) {
    pca1.setPWM(servo_index, 0, pulse);
  } else {
    pca2.setPWM(servo_index - PER_BOARD, 0, pulse);
  }
}

// ====== SETUP ======
void setup() {
  Serial.begin(115200);
  
  Wire.begin();
  Wire.setClock(400000); // 400kHz I2C
  
  // Init servos
  pca1.begin();
  pca2.begin();
  pca1.setPWMFreq(SERVO_FREQ);
  pca2.setPWMFreq(SERVO_FREQ);
  delay(100);
  
  for (int i = 0; i < SERVOS_TOTAL; i++) {
    currentAngles[i] = 90;
    writeServoChannel(i, 90);
  }
  delay(200);
  
  // Try to init IMU
  imuAvailable = initMPU9250();
  
  Serial.println("READY");
  Serial.flush();
  
  if (imuAvailable) {
    Serial.println("IMU_OK");
  } else {
    Serial.println("IMU_FAIL");
  }
  Serial.flush();
}

// ====== MAIN LOOP ======
void loop() {
  // PRIORITY 1: Process ALL servo commands IMMEDIATELY
  // This loop completes in microseconds - processes entire buffer
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    
    if (c == '\r') continue;
    
    // Ignore IMU echoes
    if (c == 'I') {
      // Flush the IMU line
      while (Serial.available() > 0 && Serial.read() != '\n');
      continue;
    }
    
    if (c == '\n') {
      rxbuf[rxidx] = '\0';
      
      // Parse CSV angles
      int vals[SERVOS_TOTAL];
      int found = 0;
      char *ptr = rxbuf;
      char *endptr;
      
      while (found < SERVOS_TOTAL && *ptr) {
        while (*ptr == ' ' || *ptr == ',') ptr++;
        if (!*ptr) break;
        
        long val = strtol(ptr, &endptr, 10);
        if (ptr == endptr) break;
        
        vals[found++] = (int)val;
        ptr = endptr;
      }
      
      // Apply if valid
      if (found == SERVOS_TOTAL) {
        bool valid = true;
        for (int i = 0; i < SERVOS_TOTAL; i++) {
          if (vals[i] < 0 || vals[i] > 180) {
            valid = false;
            break;
          }
        }
        
        if (valid) {
          for (int i = 0; i < SERVOS_TOTAL; i++) {
            currentAngles[i] = vals[i];
            writeServoChannel(i, vals[i]);
          }
        }
      }
      
      rxidx = 0;
      
    } else {
      if (rxidx < sizeof(rxbuf) - 1) {
        rxbuf[rxidx++] = c;
      } else {
        rxidx = 0;
      }
    }
  }
  
  // PRIORITY 2: IMU state machine (non-blocking)
  // Each state does ONE quick I2C read (~1ms)
  // Servos can still be processed between states!
  updateIMU();
  
  // Minimal delay
  delayMicroseconds(100);
}