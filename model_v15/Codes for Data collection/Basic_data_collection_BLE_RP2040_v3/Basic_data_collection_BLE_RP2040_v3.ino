//Date: 30 Nov 2023
//Author: Jatin_Kadge

//Following code calculates the average IMU values over 5 consecutive IMU data's and transmits/broadcasts the IMU average values in a block over BLE and send it to the Laptop at receiver end
//It also send time_stamp and counter that is incremented every time the data is sent.

//Instructions:
//After the execution of this Arduino code, please execute the following receiver code on your system. 
//Successive gesture must be performed continuously within successive 3 sec interval. Please follow the instructions given in Serial monitor.

//Recieving python code name: RP2040-Receive_via_BLE_bleak_data_collection_v2.py


#include <Arduino_LSM6DSOX.h>
#include <ArduinoBLE.h>

BLEService imuService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic imuCharacteristic("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify, 60);

unsigned long current_time;
float time_difference;
int rst_time = 3300;
unsigned long buffer_time;
unsigned long time_elapsed = 0;

int rst = 0;

int N = 5;                                  // since, average is over consecutive 5 readings
//int N = 6;                                  // since, average is over consecutive 10 readings

int debug_count = 0;
int start_count = 0; 
int scount = 0;

int count = 0;                           //a debugging variable to check data loss in transmission
int sample_count = 0;

String post_data;                        //a string variable for posting data to laptop 
String start_bit = "0";                  //a variable to indicate the start of the sent data
String end_bit = "1";                    //a variable to indicate the end of the sent data
String blank_bit = "0";

float IMU_data[7];                       //an array for IMU values, (1,2,3)=> accelerometer values and (4,5,6)=> gyroscope values_

float IMU_6N_data[7][6];                  //a 2D array to save consecutive N IMU values // 6 because it contains 3 accelerometer values and 3 gyroscope values


void setup() 
{
  Serial.begin(9600);                //Initialize serial monitor

  if (!IMU.begin()) {
        Serial.println("Device error");
        while (1);
    } 


  //firmware check code removed

  if (!BLE.begin()) {
  Serial.println("Failed to initialize BLE!");
  while (1);
  }

  BLE.setLocalName("IMU_Data - RP2040");

  BLE.setAdvertisedService(imuService);
  imuService.addCharacteristic(imuCharacteristic);
  BLE.addService(imuService);

  BLE.advertise();
  Serial.println("Bluetooth device active, waiting for connections...");
  
}


void loop()
{

  const char* resultCharArray;
  
  BLEDevice central = BLE.central();

  if (central)                                               // if you get a client,
  {
    Serial.print("Connected to central: ");
    Serial.println(central.address());

    
    ////////////////////////////////////// resetting the count //////////////////////////////////////////////////////////////////////////////////
    
    if (start_count == 0)                                   // when the client is connected for the first time
    {
      current_time = millis();                              // get the cuurent time using millis function
      time_elapsed = current_time - 0;                   // calculate the time difference to ignore the time already spent
      start_count += 1;                                      // make the start_count = 1 to run this code only once. 
    }
    /////////////////////////////////////completed////////////////////////////////////////////////////////////////////////////////////////////
    

    
    while (central.connected())                              // loop while the client's connected
    {
      if(scount == 0)
      {
        Serial.println("Please make a Gesture in next 3.3 seconds ");
        scount++;
      }

      float IMU_avg[7] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0};             // an array for average accelerometer value
      
     //////////////////////////////////////////////////    reading accelerometer and gyroscope values  /////////////////////////////////////////////////////////////////      
      for(int j=1; j<=N; j++)                                                                        //loop for saving 5 consecutive values of accelerometer into a 2D array
      {
        if (IMU.accelerationAvailable() & IMU.gyroscopeAvailable())                                    //Get accelerometer values and gyroscople values together
        {
          IMU.readAcceleration(IMU_data[1], IMU_data[2], IMU_data[3]);
          IMU.readGyroscope(IMU_data[4], IMU_data[5], IMU_data[6]);
              
          //Serial.println(++debug_count);
          delay(9);                                                                                //A delay of 9 ms is introduced to capture new data in each loop
          for(int i=1; i<=3; i++)
          {
            IMU_6N_data[i][j] = IMU_data[i];                                                      //Save 5 consecutive accelerometer values into IMU_6N_data 
            IMU_6N_data[i+3][j] = IMU_data[i+3];                                                  //Save 5 consecutive gyroscope values into IMU_6N_data
          } 
        } 
      }
      ////////////////////////////////////////////////    reading accelerometer and gyroscop value: complete ///////////////////////////////////////////////////////
 

      //////////////////////////////////////////////////              average filter                        ////////////////////////////////////////// 
      for(int i =1; i<4; i++)                                          //loop for taking an average of the accelerometer  and gyroscope values
      {
        for(int j=1; j<=N; j++)
        {
          IMU_avg[i] = IMU_avg[i] + ((IMU_6N_data[i][j])/(float)N);                                //statement taking average of Accelerometer value
          IMU_avg[i+3] = IMU_avg[i+3] + ((IMU_6N_data[i+3][j])/(float)N);      //*                      //statement taking average of Gyroscope values
        }
        if(int i = 3)
        {
          time_difference = millis() - time_elapsed;
        }
      }
      ///////////////////////////////////////////////             average flter: complete                   ////////////////////////////////////////////////


      ///////////////////////////////////////////////             sending data to laptop                  ////////////////////////////////////////////////////////

      post_data = start_bit;                                          //save start_bit = 0 in the string to be send               
      
      for(int i=1; i<7; i++)                                          //for loop for converting/saving IMU_avg[i] array values into string seperated by ','
      {
        post_data = post_data + "," + IMU_avg[i];
        
        if (i == 6)
        {
          count++;
          
          if (time_difference >= rst_time)
          {
            time_elapsed = millis();
            rst = 1;
          }
        }
      }

       

      post_data = post_data + "," + String(count) + "," + String(time_difference) + "," + end_bit;                         //save end_bit = 1 in the string to be send


      resultCharArray = post_data.c_str();
      imuCharacteristic.writeValue(reinterpret_cast<const uint8_t*>(resultCharArray), strlen(resultCharArray));

      if (rst == 1)
      {
        Serial.print("Data Collection complete for Gesture Sample ");
        Serial.println(++sample_count);
        post_data =  "0,0,0,0,0,0,0,0,0,0";
        resultCharArray = post_data.c_str();
        imuCharacteristic.writeValue(reinterpret_cast<const uint8_t*>(resultCharArray), strlen(resultCharArray));
        scount = 0;
        count = 0;
        rst = 0;
        debug_count = 0;
      }

      //////////////////////////////////////////////         sending data to laptop: complete             ////////////////////////////////////////////////////////
      
    }
  }
  //close the connection:
 Serial.println("Central server disconnected");  
}

