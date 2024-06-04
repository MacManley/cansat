//Created by Nathan Manley
//2023-1-25
//Last Edited 2023-5-04
//CanSat GCC

#include <TinyGPSPlus.h>
#include <SoftwareSerial.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_BMP280.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <SPI.h>
#include "MQ7.h"

#define A_PINCO 4 // Analogue Pin of MQ7
#define VOLTAGE 5 // Voltage of MQ7

static const int RXPin = 3, TXPin = 4;
static const uint32_t GPSBaud = 9600;

TinyGPSPlus gps;
Adafruit_MPU6050 mpu;
Adafruit_BMP280 bmp;
Adafruit_Sensor *bmp_temp = bmp.getTemperatureSensor();
Adafruit_Sensor *bmp_pressure = bmp.getPressureSensor();

SoftwareSerial ss(RXPin, TXPin);

MQ7 mq7(A_PINCO, VOLTAGE);
//height above sea lvl calculations
float p0;
float p1;
float p2;
float p3;
float p4;
float p5;
float seaLvlPress;
//raw value from each sensor every -- ms
float temp;
float hum; 
float alt;
float monoxideLvl;
//sum
//avg
//time
//int timeS;
//int timeMS;
//sample rate per second

void setup() {
  //Baud Rate of Serial port = 9600
  pinMode(7, OUTPUT);
  pinMode(6, OUTPUT);
  Serial.begin(9600);
  while(!Serial);
    delay(10);

  ss.begin(GPSBaud);
  Serial.print(F("TinyGPSPlus library v. ")); Serial.println(TinyGPSPlus::libraryVersion());
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
  }
  unsigned status;
  //status = bmp.begin(BMP280_ADDRESS_ALT, BMP280_CHIPID);
  status = bmp.begin();
  if (!status) {
    Serial.println(F("Could not find a valid BMP280 sensor, check wiring or "
                      "try a different address!"));
    Serial.print("SensorID was: 0x"); Serial.println(bmp.sensorID(),16);
    Serial.print("   ID of 0x56-0x58 represents a BMP 280\n");
    while (1) delay(10);
  }
  Serial.println("MPU6050 Found!");
  mpu.setAccelerometerRange(MPU6050_RANGE_16_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,     /* Operating Mode. */
                  Adafruit_BMP280::SAMPLING_X2,     /* Temp. oversampling */
                  Adafruit_BMP280::SAMPLING_X16,    /* Pressure oversampling */
                  Adafruit_BMP280::FILTER_X16,      /* Filtering. */
                  Adafruit_BMP280::STANDBY_MS_500); /* Standby time. */

  bmp_temp->printSensorDetails();

  Serial.println("");
  delay(100);
  Serial.println(F("Calibrating MQ7"));  
  mq7.calibrate();
  Serial.println(F("Calibration Finished"));
}
void loop() {
  Serial.print("GCC Cansat / ");
  sensors_event_t a, g, tempAda;
  sensors_event_t temp_event, pressure_event;
  mpu.getEvent(&a, &g, &tempAda);
  bmp_temp->getEvent(&temp_event);
  bmp_pressure->getEvent(&pressure_event);
  temp = readTemp(A0);
  monoxideLvl = calcMO();
  alt = calcAlt(pressure_event.pressure);
  Serial.print(F("Pressure =, "));
  Serial.print(pressure_event.pressure);
  Serial.print(F("/ Height abv Sea =, "));
  Serial.print(alt);
  Serial.print(F(" ,Height BMP =, "));
  Serial.print(bmp.readAltitude(1013.25));
  Serial.print(F(" / Temperature =, "));
  Serial.print(temp);
  Serial.print(F(" ,BMP Temp =,"));
  Serial.print(temp_event.temperature);
  Serial.print(F(" / Acceleration X:, "));
  Serial.print(a.acceleration.x);
  Serial.print(F(", Y:, "));
  Serial.print(a.acceleration.y);
  Serial.print(F(", Z:, "));
  Serial.print(a.acceleration.z);
  Serial.print(F("/ Rotation X:, "));
  Serial.print(g.gyro.x);
  Serial.print(F(", Y:, "));
  Serial.print(g.gyro.y);
  Serial.print(F(", Z:, "));
  Serial.print(g.gyro.z);
  Serial.print(F(" ,rad/s / Tem:, "));
  Serial.print(tempAda.temperature);
  Serial.print(F(" / Monoxide =, "));
  Serial.print(monoxideLvl);
  while (ss.available() > 0)
      if (gps.encode(ss.read()))
        displayInfo();
        
    if (millis() > 5000 && gps.charsProcessed() < 10)
    {
    Serial.println(F("No GPS detected: check wiring."));
    while(true);
    }
  
  if (millis() > 5000 && g.gyro.x + g.gyro.y + g.gyro.z < 2) {
    while(true);
  }
}

float readTemp(int pin) {
  int tempValue = analogRead(pin);
  int Vo = 1023 - tempValue;  //Int of volt reading
  float R = 10000;  //Fixed resistance in voltage div

  //Steinhart co-efficents  
  float c1 = 3.367853554E-03;
  float c2 = -1.070576969E-04;
  float c3 = 15.19697987E-07;
  float logRt, Rt, T;

  Rt = ((tempValue * (5.0 / 1023.0) * R) / (5.0 - (tempValue * (5.0 / 1023.0))));  //Calculate thermistor resistance
  logRt = log(Rt);

  //Apply Steinhart-Hart equation
  T = (1.0 / (c1 + c2 * logRt + c3 * logRt * logRt * logRt));

  float TCalc = KtoCelsius(T);

  return TCalc;
}

float KtoCelsius(float temp) {
  return temp - 273.15;
}

float calcAlt(float pressure) {
  seaLvlPress = 100300.0; // Update value on comp day
  p1 = pressure*100;
  p2 = p1/seaLvlPress;
  p3 = log10 (p2);
  p4 = (p3/5.25588);
  p5 = pow(10.0, p4) - 1.0;
  alt = p5/-0.0000225577;
  return alt;  
}

float calcMO() {
  if (buzzQ && isnan(mq7.readPpm()) || isinf(mq7.readPpm())) {
      return 0.00;
  } else {
  float monoxide = mq7.readPpm();
  return monoxide;
  }
}

void displayInfo()
{
  Serial.print(F("Location: ")); 
  if (gps.location.isValid())
  {
    Serial.print(gps.location.lat(), 6);
    Serial.print(F(","));
    Serial.print(gps.location.lng(), 6);
  }
  else
  {
    Serial.print(F("INVALID"));
  }

  Serial.print(F("  Date/Time: "));
  if (gps.date.isValid())
  {
    Serial.print(gps.date.month());
    Serial.print(F("/"));
    Serial.print(gps.date.day());
    Serial.print(F("/"));
    Serial.print(gps.date.year());
  }
  else
  {
    Serial.print(F("INVALID"));
  }

  Serial.print(F(" "));
  if (gps.time.isValid())
  {
    if (gps.time.hour() < 10) Serial.print(F("0"));
    Serial.print(gps.time.hour());
    Serial.print(F(":"));
    if (gps.time.minute() < 10) Serial.print(F("0"));
    Serial.print(gps.time.minute());
    Serial.print(F(":"));
    if (gps.time.second() < 10) Serial.print(F("0"));
    Serial.print(gps.time.second());
    Serial.print(F("."));
    if (gps.time.centisecond() < 10) Serial.print(F("0"));
    Serial.print(gps.time.centisecond());
  }
  else
  {
    Serial.print(F("INVALID"));
  }
  Serial.print(F(" Speed: "));
  if(gps.speed.isValid())
  {
    Serial.print(gps.speed.kmph()); 
  } else {
    Serial.print(F("INVALID"));
  }

Serial.print(F(" Altitiude: "));
  if(gps.speed.isValid())
  {
    Serial.print(gps.altitude.meters()); 
  } else {
    Serial.print(F("INVALID"));
  }  
  Serial.println();
}