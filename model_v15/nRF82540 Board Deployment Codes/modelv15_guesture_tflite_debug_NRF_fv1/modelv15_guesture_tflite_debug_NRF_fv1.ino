//Date: 05'th Jan 2023
//Author: Jatin Kadge
//Board: nrf BLE sense 
//model: model__v15


//Aim: Deployment of the ML model into nRF52840 BLE Sense - v2
//This is model with CONV_2D with 4 filters and dense layers with 10 neurons
//Tensorsize = 80
//Addded and Registered required Builtin-Ops seperately

//Tensorflow version: 2.13.0
//Tensorflow Lite library version: 2.4.0-ALPHA    //Link:https://downloads.arduino.cc/libraries/github.com/bcmi-labs/Arduino_TensorFlowLite-2.4.0-ALPHA.zip

//Error resolved:
//Error: “Didn't find op for builtin opcode 'CONV_2D' version '1'” 
//Firmware crashes after few minutes of code execution. The error is related to memory consumption by the code.


//#include <Arduino_LSM6DSOX.h>
#include "LSM6DS3.h"
#include "Wire.h"

//Create a instance of class LSM6DS3
LSM6DS3 myIMU(I2C_MODE, 0x6A);    //I2C device address 0x6A

unsigned long current_time;
float time_difference;

int N = 5;                                  // since, average is over consecutive 5 readings

float IMU_data[7];                       //an array for IMU values, (1,2,3)=> accelerometer values and (4,5,6)=> gyroscope values_
float IMU_6N_data[7][6];                 //a 2D array to save consecutive N IMU values // 6 because it contains 3 accelerometer values and 3 gyroscope values

// #undef max
// #undef min

// Import TensorFlow stuff
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
//#include "tensorflow/lite/version.h"

// Our model
#include "c_model__v15_DM2_3_3secs.h"

// Figure out what's going on in our model
#define DEBUG 1

// TFLite globals, used for compatibility with Arduino-style sketches
namespace {
  tflite::MicroErrorReporter *error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  // Create an area of memory to use for input, output, and other TensorFlow arrays. You'll need to adjust this by combiling, running, and looking for errors.
  constexpr int kTensorArenaSize = 80 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace

// array to map gesture index to a name
const char* GESTURES[] = {
  "Adduction",
  "Flexion",
  "Left Swing",
  "No motion",
  "Right Swing"
};

#define NUM_GESTURES 5
#define N_samples 90                               //a variable for sample window
#define gestures_data 6                             

float Samples[N_samples][gestures_data];
float IMU_avg[7];
float flat_array[N_samples*gestures_data];


void setup() 
{
  //tflite::InitializeTarget();

  // Wait for Serial to connect
  #if DEBUG
  while(!Serial);
  #endif

  if (myIMU.begin() == 1) {
        Serial.println("Device error");
        while (1);
  }

  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(c_model__v15_DM2_3_3secs);
  
  // pull in all the TFLM ops, you can remove this line and only pull in the TFLM ops you need, if would like to reduce the compiled size of the sketch.
  static tflite::AllOpsResolver all_ops_resolver;
  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(model, all_ops_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;


  // Allocate memory for the model's input and output tensors
  interpreter->AllocateTensors();

  // Assign model input and output buffers (tensors) to pointers
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

}

//A function to flatten the 2d array Samples [10][6] or Samples[N_samples][gestures_data]
float* flatten_2d_array(float array[N_samples][gestures_data], int rows, int cols) 
{
  float* flattened_array = new float[rows * cols];
  for (int i = 0; i < rows; i++) 
  {
    for (int j = 0; j < cols; j++) 
    {
      flattened_array[i * cols + j] = array[i][j];     
    }
  }
  return flattened_array;
}

/*
// Function to calculate available memory
int availableMemory() 
{
  int size = 237568;                              // Adjust this value based on your specific board and memory requirements; NRF Board SRAM = 270336 237568
  byte *buf;

  while ((buf = (byte *)malloc(--size)) == NULL);

  free(buf);

  return size;
}
*/


void loop()
{
  //for loop for making all the components of IMU_avg[] to be 0
  for(int i=0; i<=gestures_data; i++)
  {
    IMU_avg[i] = 0;                                               // an array for average accelerometer and gyroscope values
  }
  
  //for loop for making all the components of Samples[] to be 0
  for(int i=0; i<N_samples; i++)
  {
    for(int j=0; j<gestures_data; j++)
    {
      Samples[N_samples][gestures_data] = 0;
    }
  }    
  

  for(int k =1; k<=N_samples; k++)
  {

    for(int i=0; i<=gestures_data; i++)
  {
    IMU_avg[i] = 0;                                               // an array for average accelerometer and gyroscope values
  }
    
    //////////////////////////////////////////////////    reading accelerometer and gyroscope values  /////////////////////////////////////////////////////////////////   
    
    for(int j=1; j<=N; j++)                                                                        //loop for saving 5 consecutive values of accelerometer into a 2D array
    {
      IMU_data[1] = myIMU.readFloatAccelX();
      IMU_data[2] = myIMU.readFloatAccelY();
      IMU_data[3] = myIMU.readFloatAccelZ();
      IMU_data[4] = myIMU.readFloatGyroX();
      IMU_data[5] = myIMU.readFloatGyroY();
      IMU_data[6] = myIMU.readFloatGyroZ();
      
      for(int i=1; i<=3; i++)
      {
        IMU_6N_data[i][j] = IMU_data[i];                                                      //Save 5 consecutive accelerometer values into IMU_6N_data 
        IMU_6N_data[i+3][j] = IMU_data[i+3];                                                  //Save 5 consecutive gyroscope values into IMU_6N_data
      }
      
    }

    ////////////////////////////////////////////////    reading accelerometer and gyroscop value: complete ///////////////////////////////////////////////////////

    //////////////////////////////////////////////////              average filter                        ////////////////////////////////////////// 
    
    for(int i =1; i<4; i++)                                          //loop for taking an average of the accelerometer  and gyroscope values
    {
      for(int i=0; i<=gestures_data; i++)
      {
        IMU_avg[i] = 0;                                               // an array for average accelerometer and gyroscope values
      }

      for(int j=1; j<N; j++)
      {
        IMU_avg[i] = IMU_avg[i] + ((IMU_6N_data[i][j])/(float)N);                                //statement taking average of Accelerometer value
        IMU_avg[i+3] = IMU_avg[i+3] + ((IMU_6N_data[i+3][j])/(float)N);      //*                      //statement taking average of Gyroscope values
      }
       
      Samples[k-1][i-1] = IMU_avg[i];
      Samples[k-1][i+2] = IMU_avg[i+3];
       
    }

    ///////////////////////////////////////////////             average flter: complete                   ///////////////////////////////////////////////
  }

  //Printing Samples[][] - for debugging purpose only
  for(int k=0; k<N_samples; k++)
  {
    if(k==0)
    {
      Serial.println("Samples[10][6]");
    }
    for(int j=0; j<gestures_data; j++)
    {
      Serial.print(Samples[k][j]);
      Serial.print("\t");
    }
    Serial.print("\n");
  }
  Serial.print("\n");


  // Flatten the 2D array
  float* flattend_samples = flatten_2d_array(Samples, N_samples, gestures_data);
  
  //Copying content of flattend samples to flat_array
  memcpy(flat_array, flattend_samples, N_samples * gestures_data * sizeof(float));

  free(flattend_samples);

  //Printing flat_array - for debugging purpose only
  for(int i=0; i<N_samples*gestures_data; i++)
  {
    Serial.print(flat_array[i]);
    Serial.print("\t");
  }
  Serial.print("\n");
  Serial.print("\n");

  //Giving input to model
  for(int n=0; n < N_samples*gestures_data; n++)
  {
    Serial.print(model_input->data.f[n] = flat_array[n]);                                   //printing model input - for debugging purpose only
    Serial.print("\t");
  }
  Serial.print("\n");

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
  }

  //Printing the Output
  for (int i = 0; i < NUM_GESTURES; i++) 
  {
    Serial.print(GESTURES[i]);
    Serial.print(": ");
    Serial.println(model_output->data.f[i], 6);
    Serial.println();
  }

  //Releasing the memory uswd by *flattend samples

  /*
  //Printing the available memory
  int freeRAM = availableMemory();
  Serial.println(freeRAM);
  */

  //Giving delay for proper visibility of output - for debugging purpose only
  delay(700);
}
