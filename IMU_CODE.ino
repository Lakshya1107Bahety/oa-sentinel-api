#include <Wire.h>

// ======================================================
// ESP32 + MPU6050 BASIC IMU TEST
// ======================================================

#define SDA_PIN 21
#define SCL_PIN 22

#define MPU6050_ADDR 0x68

#define PWR_MGMT_1   0x6B
#define WHO_AM_I     0x75
#define ACCEL_XOUT_H 0x3B


void setup() {

  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("====================================");
  Serial.println("       ESP32 + MPU6050 IMU");
  Serial.println("====================================");

  // Start I2C
  Wire.begin(SDA_PIN, SCL_PIN);

  delay(100);

  // Check MPU6050
  Wire.beginTransmission(MPU6050_ADDR);
  byte error = Wire.endTransmission();

  if (error != 0) {

    Serial.println();
    Serial.println("ERROR: MPU6050 NOT DETECTED!");
    Serial.println();
    Serial.println("Check wiring:");
    Serial.println("MPU VCC -> ESP32 3.3V");
    Serial.println("MPU GND -> ESP32 GND");
    Serial.println("MPU SDA -> ESP32 GPIO 21");
    Serial.println("MPU SCL -> ESP32 GPIO 22");

    while (1) {
      delay(1000);
    }
  }

  Serial.println("MPU6050 I2C FOUND!");

  // Read WHO_AM_I
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(WHO_AM_I);
  Wire.endTransmission(false);

  Wire.requestFrom(MPU6050_ADDR, 1);

  if (Wire.available()) {

    byte whoAmI = Wire.read();

    Serial.print("WHO_AM_I = 0x");
    Serial.println(whoAmI, HEX);

  }

  // Wake up MPU6050
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(PWR_MGMT_1);
  Wire.write(0x00);
  Wire.endTransmission();

  delay(100);

  Serial.println("MPU6050 READY!");
  Serial.println();
  Serial.println("AX(g),AY(g),AZ(g),GX(deg/s),GY(deg/s),GZ(deg/s)");
}


void loop() {

  // ==============================================
  // Request 14 bytes from MPU6050
  // ==============================================

  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(ACCEL_XOUT_H);
  Wire.endTransmission(false);

  Wire.requestFrom(MPU6050_ADDR, 14);

  if (Wire.available() < 14) {

    Serial.println("Sensor read error");
    delay(100);
    return;
  }


  // ==============================================
  // ACCELEROMETER
  // ==============================================

  int16_t rawAX =
    (Wire.read() << 8) | Wire.read();

  int16_t rawAY =
    (Wire.read() << 8) | Wire.read();

  int16_t rawAZ =
    (Wire.read() << 8) | Wire.read();


  // ==============================================
  // TEMPERATURE
  // ==============================================

  Wire.read();
  Wire.read();


  // ==============================================
  // GYROSCOPE
  // ==============================================

  int16_t rawGX =
    (Wire.read() << 8) | Wire.read();

  int16_t rawGY =
    (Wire.read() << 8) | Wire.read();

  int16_t rawGZ =
    (Wire.read() << 8) | Wire.read();


  // ==============================================
  // CONVERT TO REAL UNITS
  // ==============================================

  // Accelerometer ±2g
  float ax = rawAX / 16384.0;
  float ay = rawAY / 16384.0;
  float az = rawAZ / 16384.0;

  // Gyroscope ±250 deg/s
  float gx = rawGX / 131.0;
  float gy = rawGY / 131.0;
  float gz = rawGZ / 131.0;


  // ==============================================
  // PRINT CSV
  // ==============================================

  Serial.print(ax, 4);
  Serial.print(",");

  Serial.print(ay, 4);
  Serial.print(",");

  Serial.print(az, 4);
  Serial.print(",");

  Serial.print(gx, 4);
  Serial.print(",");

  Serial.print(gy, 4);
  Serial.print(",");

  Serial.println(gz, 4);


  // ==============================================
  // 50 Hz
  // ==============================================

  delay(500);
}
