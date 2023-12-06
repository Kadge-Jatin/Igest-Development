//Date: 06'th Nov 2023
//Author: Jatin Kadge
//model: model_v15
//model c file: c_dmodel__v15_4_DM2_3_3secs_2.h

//Aim: Deployment of the ML model model_v15 into Arduino RP2040 - v1

//Tensorflow version: 2.13.0
//Tensorflow Lite library version: 1.15.0-ALPHA


#include <Arduino_LSM6DSOX.h>

unsigned long current_time;
float time_difference;

int N = 5;                                  // since, average is over consecutive 5 readings

float IMU_data[7];                       //an array for IMU values, (1,2,3)=> accelerometer values and (4,5,6)=> gyroscope values_
float IMU_6N_data[7][6];                 //a 2D array to save consecutive N IMU values // 6 because it contains 3 accelerometer values and 3 gyroscope values


// Import TensorFlow stuff
#include <TensorFlowLite.h>
#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Our model
#include "c_dmodel__v15_DM2_3_3secs_2.h"

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
  constexpr int kTensorArenaSize = 100 * 1024;
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
//#define N_samples 190                               //a variable for sample window
#define N_samples 200                                 //to create a loop for 2 secs
#define gestures_data 6                             

float Samples[N_samples][gestures_data];
float IMU_avg[7];
float flat_array[N_samples*gestures_data];


void setup() 
{

  // Wait for Serial to connect
  #if DEBUG
  while(!Serial);
  #endif

  if (!IMU.begin()) {
  Serial.println("Failed to initialize IMU!");
  while (1);
  }

  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model = tflite::GetModel(c_dmodel__v15_DM2_3_3secs_2);
  //model = tflite::GetModel(c_Nmodel__v10_3_DM2_3secs_4);

  // pull in all the TFLM ops, you can remove this line and only pull in the TFLM ops you need, if would like to reduce the compiled size of the sketch.
  tflite::MicroMutableOpResolver micro_mutable_op_resolver;
  micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_DEPTHWISE_CONV_2D, tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D, tflite::ops::micro::Register_MAX_POOL_2D());
  micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D, tflite::ops::micro::Register_CONV_2D());
  micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED, tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX, tflite::ops::micro::Register_SOFTMAX());
  micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE, tflite::ops::micro::Register_RESHAPE());
  
  // Build an interpreter to run the model
  tflite::MicroInterpreter static_interpreter(model, micro_mutable_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
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
  int size = 270336;                              // Adjust this value based on your specific board and memory requirements; Arduino Nano RP2040 SRAM = 270336
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
    
    //////////////////////////////////////////////////    reading accelerometer and gyroscope values  /////////////////////////////////////////////////////////////////   
    
    for(int j=1; j<=N; j++)                                                                        //loop for saving 5 consecutive values of accelerometer into a 2D array
    {
      if (IMU.accelerationAvailable() & IMU.gyroscopeAvailable())                                    //Get accelerometer values and gyroscople values together
      {
        IMU.readAcceleration(IMU_data[1], IMU_data[2], IMU_data[3]);
        IMU.readGyroscope(IMU_data[4], IMU_data[5], IMU_data[6]);   

        //delay(9);

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
  free(flattend_samples);

  /*
  //Printing the available memory
  int freeRAM = availableMemory();
  Serial.println(freeRAM);
  */

  //Giving delay for proper visibility of output - for debugging purpose only
  delay(700);
}







                                                       
  
  
