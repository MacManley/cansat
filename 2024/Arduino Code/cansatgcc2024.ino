//CANSAT 2024

#include <TinyGPSPlus.h>
#include <SoftwareSerial.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h> 
#include <ArduinoJson.h> 
#include <Adafruit_BME680.h> 
#include <SPI.h> 
#include <SD.h> 

static const int RXPin = 14, TXPin = 12; // GPIO14, corresponds to D5 -- for GPS // GPIO12, corresponds to D6  -- For GPS
static const uint32_t GPSBaud = 9600;
TinyGPSPlus gps;
SoftwareSerial ss(RXPin, TXPin);


static const int OpenLog_TXPin = 15; // Connect this to OpenLog's RX
static const int OpenLog_RXPin = -1; // Dummy or unused RX pin for OpenLog

SoftwareSerial openLogSerial(OpenLog_RXPin, OpenLog_TXPin);

static const int uvSensorPin = A0;

Adafruit_BME680 bme; // Initialize BME680 sensor
Adafruit_MPU6050 mpu; //Initalise MPU

unsigned long lastGSPCall = 0;  // Stores the last time GPS() was called
unsigned long lastUVCall = 0;  // Stores the last time UV() was called
unsigned long lastBMECall = 0;  // Stores the last time BME() was called
const unsigned long intervalCam = 10000; // CAMrea Interval between calls, 10 seconds = 10000 milliseconds
const unsigned long intervalGPS = 5000; // GPS Interval between calls, 5 seconds
const unsigned long intervalUV = 2500; // UV Interval between calls, 10 seconds = 10000 milliseconds
const unsigned long intervalBME = 2000; // BME Interval between calls, 5 seconds

//declarting global variable - for JSON saving
struct GlobalVariables {
  float AccelerationX;
  float AccelerationY;
  float AccelerationZ;
  float RotationX;
  float RotationY;
  float RotationZ;
  float TemperatureMPU;
  float Pressure;
  float Humidity;
  float Gas;
  float BMEtemperature;
  float UV_Voltage;
  float UV_Index;
  float Longitude;
  float Latitude;
  float datatime;
  float speed;
  float altitude;
  float AltitudeBME;
};

// globala known thing for the JSON values
GlobalVariables globalData;
//so it can be called from wihtin the viod loop
void printJSON(GlobalVariables sensorData);

void setup() {
  Serial.begin(115200); // sets baud rate
  ss.begin(GPSBaud); // start ports for for GPS

  //setting differnt serial - for use of camrea
  // Serial1.begin(19200);

  while(!Serial);

  //global variables set up
  globalData.AccelerationX = 0.0;
  globalData.AccelerationY = 0.0;
  globalData.AccelerationZ = 0.0;
  globalData.RotationX = 0.0;
  globalData.RotationY = 0.0;
  globalData.RotationZ = 0.0;
  globalData.TemperatureMPU = 0.0;
  globalData.Pressure = 0.0;
  globalData.Humidity = 0.0;
  globalData.Gas = 0.0;
  globalData.UV_Voltage = 0.0;
  globalData.UV_Index = 0.0;
  globalData.Longitude = 0.0;
  globalData.Latitude = 0.0;
  globalData.datatime = 0.0;
  globalData.speed = 0.0;
  globalData.altitude = 0.0;
  globalData.AltitudeBME = 0.0;

  //Starting ports
  Wire.begin(4, 5);

  //peramiter code for BME
  bme.setTemperatureOversampling(BME680_OS_8X);
  bme.setHumidityOversampling(BME680_OS_2X);
  bme.setPressureOversampling(BME680_OS_4X);
  bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
  bme.setGasHeater(320, 150); // 320*C for 150 ms

  //initialisation for MPU
  mpu.setAccelerometerRange(MPU6050_RANGE_16_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
}

void loop() {
  //GPS
  while (ss.available() > 0)
    if (gps.encode(ss.read()))
      GPS();

  if (millis() > 5000 && gps.charsProcessed() < 10)
  {
    Serial.println(F("No GPS detected: check wiring."));
    while(true);
  }

  //MPU -- run fast
  if (!mpu.begin()) {
    Serial.println("MPU not detected");
  } else {
    MPU();
  }

  //BME
  unsigned long currentMillisBME = millis();
  if (currentMillisBME - lastBMECall >= intervalBME) {
    if (!bme.begin(0x77, &Wire)) {
      Serial.println("BME not detected");
    } else {
      BME();
    }
    lastBMECall = currentMillisBME;
  }

  unsigned long currentMillisUV = millis();
  if (currentMillisUV - lastUVCall >= intervalUV) {
    UV();
    lastUVCall = currentMillisUV;
  }

  printJSON(globalData);
}

void MPU() {
  sensors_event_t a, g, tempAda; //declare 
  mpu.getEvent(&a, &g, &tempAda); //getting data

  //updating data to global vars
  globalData.AccelerationX = a.acceleration.x;
  globalData.AccelerationY = a.acceleration.y;
  globalData.AccelerationZ = a.acceleration.z;
  globalData.RotationX = g.gyro.x;
  globalData.RotationY = g.gyro.y;
  globalData.RotationZ = g.gyro.z;

  globalData.TemperatureMPU = tempAda.temperature;
} 

void BME() {
  if (!bme.performReading()) {
    Serial.println("Failed to perform reading");
    return;
  }

  float temperature = bme.temperature;
  globalData.BMEtemperature = temperature;

  float pressure = bme.pressure / 100.0;
  globalData.Pressure = pressure;

  float humidity = bme.humidity;
  globalData.Humidity = humidity;

  float gas = bme.gas_resistance / 1000.0;
  globalData.Gas = gas;


  float seaLevelPressure = 1014.00; // Update value on comp day
  float altitude = bme.readAltitude(seaLevelPressure);
  globalData.AltitudeBME = altitude;
}

void UV() {
  // Read the analog value from the UV sensor
  int uvValue = analogRead(uvSensorPin);

  // Convert the analog value to UV intensity in mW/cm²
  float voltage = uvValue * (3.3/1023.0);
  float reading = voltage / 0.1;

  globalData.UV_Voltage = voltage;
  globalData.UV_Index = reading;
}

void GPS() {
  while (ss.available() > 0)
    if (gps.encode(ss.read()))
      if (gps.location.isValid()) {
        float Longitude = gps.location.lat();
        globalData.Longitude = Longitude;
        float Latitude = gps.location.lng();
        globalData.Latitude = Latitude; 
      }

      if (gps.speed.isValid()) {
        float speed = gps.speed.kmph();
        globalData.speed = speed;
      }

      if (gps.altitude.isValid()) {
        float altitude = gps.altitude.meters();
        globalData.altitude = altitude;
      }

      if (gps.date.isValid()) {
        // YYYY.MMDD
        float year = gps.date.year();
        float monthDay = (gps.date.month() * 100 + gps.date.day()) / 10000.0; // Converts MMDD into a decimal
        globalData.datatime = year + monthDay; // Combines the year with the MMDD decimal
      }
}

void printJSON(GlobalVariables sensorData) {
  //reading in updated data
  float AccelerationX = globalData.AccelerationX;
  float AccelerationY = globalData.AccelerationY; 
  float AccelerationZ = globalData.AccelerationZ;
  float RotationX = globalData.RotationX;
  float RotationY = globalData.RotationY;
  float RotationZ = globalData.RotationZ;
  float TemperatureMPU = globalData.TemperatureMPU;
  float Pressure = globalData.Pressure;
  float Humidity = globalData.Humidity;
  float Gas = globalData.Gas;
  float BME_Temp = globalData.BMEtemperature;
  float UV_Voltage = globalData.UV_Voltage;
  float UV_Index = globalData.UV_Index;
  float Longitude = globalData.Longitude;
  float Latitude = globalData.Latitude;
  float datatime = globalData.datatime;
  float speed = globalData.speed;
  float altitude = globalData.altitude;
  float AltitudeBME = globalData.AltitudeBME;

  unsigned long currentTime = millis();

  StaticJsonDocument<512> jsonDoc; 

  // Populate the JSON object with data
  jsonDoc["0"] = currentTime;
  jsonDoc["1"] = sensorData.AccelerationX;
  jsonDoc["2"] = sensorData.AccelerationY;
  jsonDoc["3"] = sensorData.AccelerationZ;
  jsonDoc["4"] = sensorData.RotationX;
  jsonDoc["5"] = sensorData.RotationY;
  jsonDoc["6"] = sensorData.RotationZ;
  jsonDoc["7"] = sensorData.TemperatureMPU;
  jsonDoc["8"] = sensorData.Pressure;
  jsonDoc["9"] = sensorData.Humidity;
  jsonDoc["10"] = sensorData.Gas;
  jsonDoc["11"] = sensorData.BMEtemperature;
  jsonDoc["12"] = sensorData.UV_Voltage;
  jsonDoc["13"] = sensorData.UV_Index;
  jsonDoc["14"] = sensorData.Longitude;
  jsonDoc["15"] = sensorData.Latitude;
  jsonDoc["16"] = sensorData.datatime;
  jsonDoc["17"] = sensorData.speed;
  jsonDoc["18"] = sensorData.altitude;
  jsonDoc["19"] = sensorData.AltitudeBME;

  String jsonString;
  serializeJson(jsonDoc, jsonString);

  Serial.println(jsonString);
}