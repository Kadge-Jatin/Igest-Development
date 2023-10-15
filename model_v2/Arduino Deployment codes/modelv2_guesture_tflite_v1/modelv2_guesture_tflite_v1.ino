//03rd OCT 2023
//Author: Jatin Kadge
//Aim: Deployment of the ML model into Arduino RP2040


//Tensorflow version: 2.13.0
//Tensorflow Lite library version: 2.4.0-ALPHA

//Error faced:
//Error 1: Unable to push 2D array as the input to model
//Error 2: Didn't find op for builtincode 'EXPAND_DIMS' version'1', Failed to get registration from op code EXPAND_DIMS, Failed started model allocation


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
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>


// Our model
#include "c_model_v2.h"

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
  constexpr int kTensorArenaSize = 5 * 1024;
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


int N_samples = 10;                                //a variable for sample window
float Samples[11][6];


void setup() 
{

// Wait for Serial to connect
#if DEBUG
  while(!Serial);
#endif

  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model = tflite::GetModel(c_model_v2);
  if (model->version() != TFLITE_SCHEMA_VERSION) 
  {
    Serial.println("Model schema mismatch!");
    while(1);
  }

  // pull in all the TFLM ops. You can remove this line and only pull in the TFLM ops you need, if would like to reduce the compiled size of the sketch.
  tflite::AllOpsResolver tflOpsResolver;

  // Build an interpreter to run the model
  interpreter = new tflite::MicroInterpreter(model, tflOpsResolver, tensor_arena, kTensorArenaSize, error_reporter);

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
  Serial.print("Input type: ");
  Serial.println(model_input->type);
#endif
*/
} 

void loop()
{
  
  float IMU_avg[7] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0};                                                 // an array for average accelerometer and gyroscope values
  float Samples[11][6] = {0};


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
       
       Samples[k][i] = IMU_avg[i];
       Samples[k][i+3] = IMU_avg[i+3];
     }

     ///////////////////////////////////////////////             average flter: complete                   ///////////////////////////////////////////////
  }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Following are the set of codes I tried to push a 2D aray into model
//However both failed
//1.

  for(int k=0; k<N_samples; k++)
  {
    for(int i=0; i<6; i++)
    {
      model_input->data.f[k][i] = Samples[k][i];
    }
  }

//2.
  model_input->data.f[0] = Samples;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  for (int i = 0; i < NUM_GESTURES; i++) 
  {
    Serial.print(GESTURES[i]);
    Serial.print(": ");
    Serial.println(model_output->data.f[i], 6);
    Serial.println();
  }
}
