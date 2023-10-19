//Author: Jatin Kadge
//model: model_v8
//Aim: Deployment of the ML model into Arduino RP2040 - v1
//This is model with 16 filters

//Tensorflow version: 2.13.0
//Tensorflow Lite library version: 1.15.0-ALPHA

//Error resolved:
//Error: Unable to push 2D array as the input to model
//Error: Didn't find op for builtincode 'EXPAND_DIMS' version'1', Failed to get registration from op code EXPAND_DIMS, Failed started model allocation
//tensorflow\lite\micro\kernels\cmsis-nn\fully_connected.cpp Hybrid models are not supported on TFLite Micro. Node FULLY_CONNECTED (number 4f) failed to prepare with status 1
//Firmware Crashing as the code is executed

//Error faced:
//The Firmware on RP2040 crashed due to the following error: “Didn't find op for builtin opcode 'CONV_2D' version '1'” 


#include <Arduino_LSM6DSOX.h>

unsigned long current_time;
float time_difference;

int N = 5;                                  // since, average is over consecutive 5 readings

float IMU_data[7];                       //an array for IMU values, (1,2,3)=> accelerometer values and (4,5,6)=> gyroscope values_
//
float IMU_6N_data[7][6];                  //a 2D array to save consecutive N IMU values // 6 because it contains 3 accelerometer values and 3 gyroscope values
//

// Import TensorFlow stuff
#include <TensorFlowLite.h>
#include <tensorflow/lite/experimental/micro/micro_error_reporter.h>
#include <tensorflow/lite/experimental/micro/kernels/micro_ops.h>
#include <tensorflow/lite/experimental/micro/micro_interpreter.h>
#include <tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/version.h>

// Our model
#include "c_model_v8.h"

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
  constexpr int kTensorArenaSize = 24 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace


// array to map gesture index to a name
const char* GESTURES[] = {
  "NO motion",
  "Left Swing",
  "Right Swing",
  "Adduction",
  "Flexion"
};

#define NUM_GESTURES 5

#define N_samples 10                               //a variable for sample window
#define gestures_data 6
float Samples[N_samples][gestures_data];


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
  model = tflite::GetModel(c_model_v8);

  // pull in all the TFLM ops, you can remove this line and only pull in the TFLM ops you need, if would like to reduce the compiled size of the sketch.
  tflite::MicroMutableOpResolver micro_mutable_op_resolver;
  micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED, tflite::ops::micro::Register_FULLY_CONNECTED(), 1, 3);

  // Build an interpreter to run the model
  tflite::MicroInterpreter static_interpreter(model, micro_mutable_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for the model's input and output tensors
  interpreter->AllocateTensors();

 // Assign model input and output buffers (tensors) to pointers
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  // Get information about the memory area to use for the model's input
  // Supported data types:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h#L226

/*
#if DEBUG
  Serial.print("Number of dimensions: ");
  Serial.println(model_input->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(model_input->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(model_input->dims->data[1]);
  Serial.print("Dim 3 size: ");
  Serial.println(model_input->dims->data[2]);
    Serial.print("Dim 4 size: ");
  Serial.println(model_input->dims->data[3]);
  Serial.print("Input type: ");
  Serial.println(model_input->type);
#endif
*/
} 

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

void loop()
{

  float IMU_avg[7] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0};                                                 // an array for average accelerometer and gyroscope values
  float Samples[N_samples][gestures_data] = {0};
  
  for(int k =1; k<=N_samples; k++)
  {
    
    //////////////////////////////////////////////////    reading accelerometer and gyroscope values  /////////////////////////////////////////////////////////////////   
    
    for(int j=1; j<=N; j++)                                                                        //loop for saving 5 consecutive values of accelerometer into a 2D array
    {
      if (IMU.accelerationAvailable() & IMU.gyroscopeAvailable())                                    //Get accelerometer values and gyroscople values together
      {
        IMU.readAcceleration(IMU_data[1], IMU_data[2], IMU_data[3]);
        IMU.readGyroscope(IMU_data[4], IMU_data[5], IMU_data[6]);   

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

  float flat_array[N_samples*gestures_data];
  
  memcpy(flat_array, flattend_samples, N_samples * gestures_data * sizeof(float));

  for(int i=0; i<N_samples*gestures_data; i++)
  {
    Serial.print(flat_array[i]);
    Serial.print("\t");
  }

  Serial.print("\n");
  Serial.print("\n");

  for(int n=0; n < N_samples*gestures_data; n++)
  {
    //model_input->data.f[n] = flat_array[n];
    Serial.print(model_input->data.f[n] = flat_array[n]);
    Serial.print("\t");
  }

  Serial.print("\n");

  // Run inference
  
  TfLiteStatus invoke_status = interpreter->Invoke();
  
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
  }

  for (int i = 0; i < NUM_GESTURES; i++) 
  {
    Serial.print(GESTURES[i]);
    Serial.print(": ");
    Serial.println(model_output->data.f[i], 6);
    Serial.println();
  }

}


/*
Output on Uploading:

Samples[10][6]
-0.03	-0.79	-0.15	-0.07	0.07	-0.16	
-0.07	-1.58	-0.31	-0.22	0.17	-0.38	
-0.10	-2.37	-0.46	-0.37	0.27	-0.60	
-0.13	-3.16	-0.61	-0.60	0.37	-0.72	
-0.17	-3.95	-0.77	-0.83	0.46	-0.84	
-0.20	-4.74	-0.92	-0.95	0.56	-0.95	
-0.23	-5.54	-1.07	-1.07	0.66	-1.06	
-0.27	-6.33	-1.23	-1.50	0.76	-1.04	
-0.30	-7.12	-1.38	-1.93	0.85	-1.01	
-0.34	-7.91	-1.53	-2.22	0.95	-0.96	

-0.03	-0.79	-0.15	-0.07	0.07	-0.16	-0.07	-1.58	-0.31	-0.22	0.17	-0.38	-0.10	-2.37	-0.46	-0.37	0.27	-0.60	-0.13	-3.16	-0.61	-0.60	0.37	-0.72	-0.17	-3.95	-0.77	-0.83	0.46	-0.84	-0.20	-4.74	-0.92	-0.95	0.56	-0.95	-0.23	-5.54	-1.07	-1.07	0.66	-1.06	-0.27	-6.33	-1.23	-1.50	0.76	-1.04	-0.30	-7.12	-1.38	-1.93	0.85	-1.01	-0.34	-7.91	-1.53	-2.22	0.95	-0.96	

-0.03	-0.79	-0.15	-0.07	0.07	-0.16	-0.07	-1.58	-0.31	-0.22	0.17	-0.38	-0.10	-2.37	-0.46	-0.37	0.27	-0.60	-0.13	-3.16	-0.61	-0.60	0.37	-0.72	-0.17	-3.95	-0.77	-0.83	0.46	-0.84	-0.20	-4.74	-0.92	-0.95	0.56	-0.95	-0.23	-5.54	-1.07	-1.07	0.66	-1.06	-0.27	-6.33	-1.23	-1.50	0.76	-1.04	-0.30	-7.12	-1.38	-1.93	0.85	-1.01	-0.34	-7.91	-1.53	-2.22	0.95	-0.96	
Didn't find op for builtin opcode 'CONV_2D' version '1'

Invoke failed!
NO motion: 0.000000

Left Swing: 0.000000

Right Swing: 0.000000

Adduction: 0.000000

Flexion: 0.000000

------------------------Firmware Crashes

*/