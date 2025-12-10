#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ===================== PCA9685 DRIVERS =====================
Adafruit_PWMServoDriver pca1 = Adafruit_PWMServoDriver(0x40); // First PCA9685
Adafruit_PWMServoDriver pca2 = Adafruit_PWMServoDriver(0x41); // Second PCA9685

// ===================== SERVO PULSE LIMITS ==================
#define SERVOMIN 110
#define SERVOMAX 590
#define SERVO_FREQ 50     // 50 Hz for most hobby servos

// Set to 1 if you want spammy debug for every joint move
#define VERBOSE_DEBUG 0

// ===================== LEG / JOINT NAMES ===================
enum Leg  : uint8_t { FL=0, FR=1, RL=2, RR=3 };   // Front-Left, Front-Right, Rear-Left, Rear-Right
enum Joint: uint8_t { HIP=0, KNEE=1, ANKLE=2 };   // 3 joints per leg

// ===================== SERVO MAPPING =======================
struct ServoID {
  uint8_t board;    // 0 -> pca1, 1 -> pca2
  uint8_t channel;  // 0..15
  int8_t  trim;     // degrees (+ CW, - CCW)
  bool    reversed; // true => angle command is mirrored around 90°
};

// 4 legs x 3 joints = 12 entries — EDIT to match your wiring if needed.
ServoID mapTable[4][3] = {
  /* FL */ { {0,0, 0,false}, {0,1, 0,false}, {0,2, 0,false} },       // HIP,KNEE,ANKLE
  /* FR */ { {1,3, 0,false}, {1,4, 0,false}, {1,5, 0,false}  },
  /* RL */ { {0,3, 0,false}, {0,4, 0,false}, {0,5, 0,false} },
  /* RR */ { {1,0, 0,false}, {1,1, 0,false}, {1,2, 0,false}  }
};

// ===================== ANGLE → PULSE =======================
int angleToPulseCalibrated(int angleDeg, const ServoID& s) {
  int a = constrain(angleDeg, 0, 180);
  if (s.reversed) a = 180 - a;
  a = constrain(a + s.trim, 0, 180);
  int commandedAngle = map(a, 0, 180, -5, 175);  // small deadband near ends
  return map(commandedAngle, 0, 180, SERVOMIN, SERVOMAX);
}

// ===================== LOW-LEVEL SET =======================
void setServoPulse(const ServoID& s, int pulse) {
  if (s.board == 0) {
    pca1.setPWM(s.channel, 0, pulse);
  } else {
    pca2.setPWM(s.channel, 0, pulse);
  }
}

// ===================== HIGH-LEVEL HELPERS ==================
void setJointAngle(Leg leg, Joint joint, int angleDeg) {
  const ServoID& s = mapTable[(uint8_t)leg][(uint8_t)joint];
  int pulse = angleToPulseCalibrated(angleDeg, s);
  setServoPulse(s, pulse);

#if VERBOSE_DEBUG
  Serial.print("Leg ");
  switch(leg){
    case FL: Serial.print("FL"); break;
    case FR: Serial.print("FR"); break;
    case RL: Serial.print("RL"); break;
    case RR: Serial.print("RR"); break;
  }
  Serial.print(" | Joint ");
  switch(joint){
    case HIP:   Serial.print("HIP");   break;
    case KNEE:  Serial.print("KNEE");  break;
    case ANKLE: Serial.print("ANKLE"); break;
  }
  Serial.print(" -> "); Serial.print(angleDeg);
  Serial.print("° (pulse "); Serial.print(pulse); Serial.println(")");
#endif
}

void setLegAngles(Leg leg, int hipDeg, int kneeDeg, int ankleDeg) {
  setJointAngle(leg, HIP,   hipDeg);
  setJointAngle(leg, KNEE,  kneeDeg);
  setJointAngle(leg, ANKLE, ankleDeg);
}

void setAllNeutral(int angleDeg = 90) {
  for (uint8_t L = 0; L < 4; L++) {
    for (uint8_t J = 0; J < 3; J++) {
      setJointAngle((Leg)L, (Joint)J, angleDeg);
      delay(10);
    }
  }
}

// ===================== GAIT TABLES (ANGLES) =====================
// Each gait is: steps x 4 legs x 3 joints (HIP,KNEE,ANKLE)

// Forward gait
const int gaitForward[][4][3] = {
  #include "spid_gait_forward.h"
};
const size_t NUM_STEPS_FORWARD = sizeof(gaitForward) / sizeof(gaitForward[0]);

// Backward gait
const int gaitBackward[][4][3] = {
  #include "spid_gait_backwards.h"
};
const size_t NUM_STEPS_BACKWARD = sizeof(gaitBackward) / sizeof(gaitBackward[0]);

// Right crawl gait
const int gaitRight[][4][3] = {
  #include "spid_gait_right.h"
};
const size_t NUM_STEPS_RIGHT = sizeof(gaitRight) / sizeof(gaitRight[0]);

// Left crawl gait
const int gaitLeft[][4][3] = {
  #include "spid_gait_left.h"
};
const size_t NUM_STEPS_LEFT = sizeof(gaitLeft) / sizeof(gaitLeft[0]);

// Yaw right (rotate in place CW) gait
const int gaitYawRight[][4][3] = {
  #include "spid_gait_yaw_right.h"
};
const size_t NUM_STEPS_YAW_RIGHT = sizeof(gaitYawRight) / sizeof(gaitYawRight[0]);

// Yaw left (rotate in place CCW) gait
const int gaitYawLeft[][4][3] = {
  #include "spid_gait_yaw_left.h"
};
const size_t NUM_STEPS_YAW_LEFT = sizeof(gaitYawLeft) / sizeof(gaitYawLeft[0]);

// ===================== HOME / NEUTRAL POSE =====================
const int homePose[4][3] = {
  { 90, 35, 35 },   // FL: HIP, KNEE, ANKLE
  { 90, 35, 35 },   // FR
  { 90, 35, 35 },   // RL
  { 90, 35, 35 }    // RR
};

void setHomePose() {
  for (uint8_t leg = 0; leg < 4; leg++) {
    setLegAngles(
      (Leg)leg,
      homePose[leg][HIP],
      homePose[leg][KNEE],
      homePose[leg][ANKLE]
    );
  }
}

// ===================== GAIT RUNTIME STRUCTURE =====================
struct GaitDef {
  const int (*steps)[4][3];  // pointer to [4][3] step array
  size_t numSteps;
};

enum GaitIndex : uint8_t {
  GAIT_FWD       = 0,
  GAIT_BACK      = 1,
  GAIT_RIGHT     = 2,
  GAIT_LEFT      = 3,
  GAIT_YAW_RIGHT = 4,
  GAIT_YAW_LEFT  = 5,
  GAIT_COUNT     = 6
};

// Map gait index to actual tables
GaitDef gaitDefs[GAIT_COUNT] = {
  { gaitForward,   NUM_STEPS_FORWARD   },
  { gaitBackward,  NUM_STEPS_BACKWARD  },
  { gaitRight,     NUM_STEPS_RIGHT     },
  { gaitLeft,      NUM_STEPS_LEFT      },
  { gaitYawRight,  NUM_STEPS_YAW_RIGHT },
  { gaitYawLeft,   NUM_STEPS_YAW_LEFT  }
};

// Current gait state
int8_t        activeGaitIndex = -1;   // -1 = STOP / home
size_t        currentStep     = 0;
unsigned long lastStepTime    = 0;
unsigned long stepDelayMs     = 1500;  // approx total time per step transition

// Smoothing: number of blended frames between steps (like your smoothSteps)
uint8_t smoothSteps = 20;             // 0 = no smoothing, 10–20 = smooth
uint8_t blendPhase  = 0;              // 0..smoothSteps

// ===================== UTILITY: APPLY ONE STEP =====================
void applyStep(const int step[4][3]) {
  for (uint8_t leg = 0; leg < 4; leg++) {
    setLegAngles(
      (Leg)leg,
      step[leg][HIP],
      step[leg][KNEE],
      step[leg][ANKLE]
    );
  }
}

// Blended pose between fromIdx and toIdx, mix in [0,255]
void applyBlendedPose(const int (*steps)[4][3],
                      size_t fromIdx, size_t toIdx, uint8_t mix) {
  for (uint8_t leg = 0; leg < 4; leg++) {
    int hip   = map(mix, 0, 255, steps[fromIdx][leg][HIP],   steps[toIdx][leg][HIP]);
    int knee  = map(mix, 0, 255, steps[fromIdx][leg][KNEE],  steps[toIdx][leg][KNEE]);
    int ankle = map(mix, 0, 255, steps[fromIdx][leg][ANKLE], steps[toIdx][leg][ANKLE]);
    setLegAngles((Leg)leg, hip, knee, ankle);
  }
}

// Smoothly blend from one 4x3 pose to another.
// transSmoothSteps: how many blend frames (use smoothSteps or a custom number)
// totalMs: total transition time in milliseconds
void smoothTransitionPose(const int fromPose[4][3],
                          const int toPose[4][3],
                          uint8_t transSmoothSteps,
                          unsigned long totalMs)
{
  if (transSmoothSteps == 0) {
    // Just jump to new pose
    for (uint8_t leg = 0; leg < 4; leg++) {
      setLegAngles((Leg)leg,
                   toPose[leg][HIP],
                   toPose[leg][KNEE],
                   toPose[leg][ANKLE]);
    }
    return;
  }

  unsigned long frameMs = totalMs / (unsigned long)(transSmoothSteps + 1);
  if (frameMs == 0) frameMs = 1;

  for (uint8_t b = 0; b <= transSmoothSteps; b++) {
    uint8_t mix = map(b, 0, transSmoothSteps, 0, 255);
    for (uint8_t leg = 0; leg < 4; leg++) {
      int hip   = map(mix, 0, 255, fromPose[leg][HIP],   toPose[leg][HIP]);
      int knee  = map(mix, 0, 255, fromPose[leg][KNEE],  toPose[leg][KNEE]);
      int ankle = map(mix, 0, 255, fromPose[leg][ANKLE], toPose[leg][ANKLE]);
      setLegAngles((Leg)leg, hip, knee, ankle);
    }
    delay(frameMs);   // blocking transition, just during gait switch / stop
  }
}

// ===================== SERIAL CONTROL =====================
const char* gaitNameFromIndex(int8_t idx) {
  switch (idx) {
    case GAIT_FWD:       return "FORWARD";
    case GAIT_BACK:      return "BACKWARD";
    case GAIT_RIGHT:     return "RIGHT";
    case GAIT_LEFT:      return "LEFT";
    case GAIT_YAW_RIGHT: return "YAW_RIGHT";
    case GAIT_YAW_LEFT:  return "YAW_LEFT";
    default:             return "NONE";
  }
}

void setActiveGait(int8_t idx) {
  // STOP → go to home pose (with smooth transition)
  if (idx < 0 || idx >= (int8_t)GAIT_COUNT) {
    Serial.println(F("[GAIT] STOP / HOME"));

    int fromPose[4][3];

    if (activeGaitIndex >= 0) {
      // build fromPose from current gait / step
      GaitDef &gFrom = gaitDefs[activeGaitIndex];
      const int (*stepsFrom)[4][3] = gFrom.steps;
      size_t stepIdx = currentStep;
      if (stepIdx >= gFrom.numSteps) stepIdx = 0;
      for (uint8_t leg = 0; leg < 4; leg++) {
        fromPose[leg][HIP]   = stepsFrom[stepIdx][leg][HIP];
        fromPose[leg][KNEE]  = stepsFrom[stepIdx][leg][KNEE];
        fromPose[leg][ANKLE] = stepsFrom[stepIdx][leg][ANKLE];
      }
    } else {
      // assume we are already at home
      for (uint8_t leg = 0; leg < 4; leg++) {
        fromPose[leg][HIP]   = homePose[leg][HIP];
        fromPose[leg][KNEE]  = homePose[leg][KNEE];
        fromPose[leg][ANKLE] = homePose[leg][ANKLE];
      }
    }

    smoothTransitionPose(fromPose, homePose, smoothSteps, 400); // 400 ms stop transition

    activeGaitIndex = -1;
    currentStep     = 0;
    blendPhase      = 0;
    lastStepTime    = millis();
    return;
  }

  // Same gait → do nothing
  if (idx == activeGaitIndex) {
    return;
  }

  // Build fromPose (current pose: either current gait or home)
  int fromPose[4][3];
  if (activeGaitIndex >= 0) {
    GaitDef &gFrom = gaitDefs[activeGaitIndex];
    const int (*stepsFrom)[4][3] = gFrom.steps;
    size_t stepIdx = currentStep;
    if (stepIdx >= gFrom.numSteps) stepIdx = 0;
    for (uint8_t leg = 0; leg < 4; leg++) {
      fromPose[leg][HIP]   = stepsFrom[stepIdx][leg][HIP];
      fromPose[leg][KNEE]  = stepsFrom[stepIdx][leg][KNEE];
      fromPose[leg][ANKLE] = stepsFrom[stepIdx][leg][ANKLE];
    }
  } else {
    // coming from home pose
    for (uint8_t leg = 0; leg < 4; leg++) {
      fromPose[leg][HIP]   = homePose[leg][HIP];
      fromPose[leg][KNEE]  = homePose[leg][KNEE];
      fromPose[leg][ANKLE] = homePose[leg][ANKLE];
    }
  }

  // Build toPose = step 0 of new gait
  int toPose[4][3];
  GaitDef &gTo = gaitDefs[idx];
  const int (*stepsTo)[4][3] = gTo.steps;
  size_t toStepIdx = 0;
  for (uint8_t leg = 0; leg < 4; leg++) {
    toPose[leg][HIP]   = stepsTo[toStepIdx][leg][HIP];
    toPose[leg][KNEE]  = stepsTo[toStepIdx][leg][KNEE];
    toPose[leg][ANKLE] = stepsTo[toStepIdx][leg][ANKLE];
  }

  // Smooth transition between gaits (tune 400 ms if needed)
  smoothTransitionPose(fromPose, toPose, smoothSteps, 400);

  // Now commit the new gait
  activeGaitIndex = idx;
  currentStep     = 0;
  blendPhase      = 0;
  lastStepTime    = millis(); // continue from here

  Serial.print(F("[GAIT] Switched to "));
  Serial.println(gaitNameFromIndex(idx));
}

void handleSerialCommands() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    switch (c) {
      case 'w': // forward
      case 'W':
        setActiveGait(GAIT_FWD);
        break;

      case 's': // backward
      case 'S':
        setActiveGait(GAIT_BACK);
        break;

      case 'd': // right
      case 'D':
        setActiveGait(GAIT_RIGHT);
        break;

      case 'a': // left
      case 'A':
        setActiveGait(GAIT_LEFT);
        break;

      case 'e': // yaw right
      case 'E':
        setActiveGait(GAIT_YAW_RIGHT);
        break;

      case 'q': // yaw left
      case 'Q':
        setActiveGait(GAIT_YAW_LEFT);
        break;

      case 'x': // stop / home
      case 'X':
        setActiveGait(-1);
        break;

      case '+': // speed up (shorter overall step time)
        if (stepDelayMs > 100) stepDelayMs -= 50;
        Serial.print(F("[GAIT] Faster, stepDelayMs = "));
        Serial.println(stepDelayMs);
        break;

      case '-': // slow down (longer overall step time)
        stepDelayMs += 50;
        Serial.print(F("[GAIT] Slower, stepDelayMs = "));
        Serial.println(stepDelayMs);
        break;

      default:
        // ignore others; you can add help here if you want
        break;
    }
  }
}

// ===================== GAIT STEPPER (NON-BLOCKING + SMOOTH) =====================
void stepGaitIfNeeded() {
  static int8_t lastGaitIndex = -2;   // something impossible at start

  // If no gait active → hold home pose
  if (activeGaitIndex < 0) {
    if (lastGaitIndex != -1) {
      Serial.println(F("[GAIT] Holding HOME pose."));
      lastGaitIndex = -1;
    }
    return;
  }

  // Get selected gait
  GaitDef &g = gaitDefs[activeGaitIndex];

  // If gait changed since last time, reset step counter
  if (lastGaitIndex != activeGaitIndex) {
    Serial.print(F("[GAIT] Starting gait "));
    Serial.print(gaitNameFromIndex(activeGaitIndex));
    Serial.print(F(" with "));
    Serial.print(g.numSteps);
    Serial.println(F(" steps."));
    currentStep  = 0;
    blendPhase   = 0;
    lastStepTime = 0;   // trigger immediate step
    lastGaitIndex = activeGaitIndex;
  }

  // Timing: if smoothing is enabled, we break stepDelayMs into (smoothSteps+1) frames
  unsigned long now = millis();
  unsigned long frameInterval;
  if (smoothSteps == 0) {
    frameInterval = stepDelayMs;
  } else {
    frameInterval = stepDelayMs / (unsigned long)(smoothSteps + 1);
    if (frameInterval == 0) frameInterval = 1;
  }

  if (now - lastStepTime < frameInterval) {
    return; // not time yet
  }
  lastStepTime = now;

  const int (*steps)[4][3] = g.steps;
  size_t nextStep = (currentStep + 1) % g.numSteps;

  if (smoothSteps == 0) {
    // No smoothing: just jump between steps
    applyStep(steps[currentStep]);
    currentStep = nextStep;
    return;
  }

  // With smoothing:
  if (blendPhase == 0) {
    // Initial hold at the exact pose
    applyStep(steps[currentStep]);
    blendPhase = 1;
  } else if (blendPhase <= smoothSteps) {
    // Blend from currentStep to nextStep
    uint8_t mix = map(blendPhase, 0, smoothSteps, 0, 255);
    applyBlendedPose(steps, currentStep, nextStep, mix);
    blendPhase++;
  }

  // After finishing blend, advance to next step
  if (blendPhase > smoothSteps) {
    currentStep = nextStep;
    blendPhase  = 0;
  }
}

// ===================== IMU SECTION (MPU9250 + AK8963) =====================

// MPU9250 and AK8963 I2C addresses
#define MPU9250_ADDR  0x68  // Default address (or 0x69 if AD0 is high)
#define AK8963_ADDR   0x0C

// MPU9250 registers
#define MPU9250_PWR_MGMT_1   0x6B
#define MPU9250_CONFIG       0x1A
#define MPU9250_GYRO_CONFIG  0x1B
#define MPU9250_ACCEL_CONFIG 0x1C
#define MPU9250_INT_PIN_CFG  0x37
#define MPU9250_WHO_AM_I     0x75

// AK8963 registers
#define AK8963_CNTL1         0x0A

// IMU state machine
enum IMUState {
  IMU_IDLE,
  IMU_READ_ACCEL,
  IMU_READ_GYRO,
  IMU_READ_MAG,
  IMU_PUBLISH
};

IMUState imuState = IMU_IDLE;
unsigned long lastIMUTime = 0;
const int IMU_INTERVAL = 50; // 50 ms = 20 Hz
bool imuAvailable = false;

// Raw IMU data
int16_t ax, ay, az, gx, gy, gz, mx, my, mz;

// I2C helpers
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

// IMU init
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
  writeByte(MPU9250_ADDR, MPU9250_INT_PIN_CFG, 0x02);    // Enable magnetometer passthrough

  // Init magnetometer
  writeByte(AK8963_ADDR, AK8963_CNTL1, 0x00);
  delay(10);
  writeByte(AK8963_ADDR, AK8963_CNTL1, 0x16); // 16-bit, 100Hz
  delay(10);

  return true;
}

// Non-blocking IMU update
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
      // Quick read: accel (6 bytes)
      uint8_t rawData[6];
      readBytes(MPU9250_ADDR, 0x3B, 6, rawData);
      ax = ((int16_t)rawData[0] << 8) | rawData[1];
      ay = ((int16_t)rawData[2] << 8) | rawData[3];
      az = ((int16_t)rawData[4] << 8) | rawData[5];
      imuState = IMU_READ_GYRO;
      break;
    }

    case IMU_READ_GYRO: {
      // Quick read: gyro (6 bytes)
      uint8_t rawData[6];
      readBytes(MPU9250_ADDR, 0x43, 6, rawData);
      gx = ((int16_t)rawData[0] << 8) | rawData[1];
      gy = ((int16_t)rawData[2] << 8) | rawData[3];
      gz = ((int16_t)rawData[4] << 8) | rawData[5];
      imuState = IMU_READ_MAG;
      break;
    }

    case IMU_READ_MAG: {
      // Quick read: magnetometer (6 bytes)
      uint8_t rawData[6];
      readBytes(AK8963_ADDR, 0x03, 6, rawData);
      mx = ((int16_t)rawData[1] << 8) | rawData[0];
      my = ((int16_t)rawData[3] << 8) | rawData[2];
      mz = ((int16_t)rawData[5] << 8) | rawData[4];
      imuState = IMU_PUBLISH;
      break;
    }

    case IMU_PUBLISH: {
      // Convert raw values to physical units
      float ax_ms2 = (ax / 16384.0f) * 9.81f;
      float ay_ms2 = (ay / 16384.0f) * 9.81f;
      float az_ms2 = (az / 16384.0f) * 9.81f;

      float gx_rads = (gx / 131.0f) * (PI / 180.0f);
      float gy_rads = (gy / 131.0f) * (PI / 180.0f);
      float gz_rads = (gz / 131.0f) * (PI / 180.0f);

      // Stream IMU data over Serial in CSV format
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

// ===================== SETUP / LOOP ========================
void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  Serial.println(F("=== Spider Robot Gait Controller + IMU ==="));
  Serial.println(F("Controls (send over serial):"));
  Serial.println(F("  w = forward"));
  Serial.println(F("  s = backward"));
  Serial.println(F("  d = right"));
  Serial.println(F("  a = left"));
  Serial.println(F("  e = yaw right (rotate CW)"));
  Serial.println(F("  q = yaw left  (rotate CCW)"));
  Serial.println(F("  x = stop / home"));
  Serial.println(F("  + / - = faster / slower"));
  Serial.println(F("  smoothSteps is set in code (default 20)."));

  // I2C init for PCA9685 + IMU
  Wire.begin();
  Wire.setClock(400000); // 400 kHz I2C

  pca1.begin();
  pca2.begin();
  pca1.setPWMFreq(SERVO_FREQ);
  pca2.setPWMFreq(SERVO_FREQ);
  delay(500);

  Serial.println(F("Moving all joints to HOME pose..."));
  setHomePose();
  delay(500);

  // Initialize IMU
  imuAvailable = initMPU9250();
  if (imuAvailable) {
    Serial.println(F("IMU_OK"));
  } else {
    Serial.println(F("IMU_FAIL"));
  }

  Serial.println(F("Ready for gait commands."));
  activeGaitIndex = -1;
  currentStep     = 0;
  blendPhase      = 0;
  lastStepTime    = millis();
}

void loop() {
  handleSerialCommands();  // read user commands (headless control)
  stepGaitIfNeeded();      // advance current gait if it's time (with smoothing)
  updateIMU();             // non-blocking IMU state machine
}
