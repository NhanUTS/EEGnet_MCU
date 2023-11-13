#include <Arduino.h>  //comment out line 63, weird conflict redefinition of abs()
                      //might be standard/compatibility/optimisation related 

// #include <stdio.h>
// #include <stdlib.h>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"

// #include "EEGnet.h"
// #include "EEGnet_model.h"
// #include "main_functions.h" //name-mangling
#include "test_model/NeuralNetwork.h"
#include "test_model/model_data.h"

#include <SDRAM.h>

#include "SDMMCBlockDevice.h"
#include "FATFileSystem.h"



// For SDram
#define SDRAM_ARRAY_BYTES 8000000 // give 8MB of SDRAM
#define SDRAM_FILL_VALUE 0x55 // value to fill SDRAM array with
#define SDRAM_MAX 8388607 //2^23

SDRAMClass SDram;
// volatile uint8_t* SDaddress;
uint8_t* SDaddress;
uint8_t* RAMptr ; 
// SDRAMClass* SDRAMptr = &SDram;

// For SD card
SDMMCBlockDevice block_device;
mbed::FATFileSystem fs("fs");

// EEGnet *eegNET;

NeuralNetwork *nn;
bool test = true;  

namespace 
{
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  int inference_count = 0;


    //for nn model
  using NN_OpResolver = tflite::MicroMutableOpResolver<7>; //important!! - put min 10, eventhough there's
                                                                //only 7 operators

  TfLiteStatus RegisterOps(NN_OpResolver& op_resolver) 
  {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());

  return kTfLiteOk;
  }
}  // namespace

void sdMounting_def (void);
void setup()
{
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);

  Serial.begin(115200);
  delay(3000);
  Serial.println("13680119");
  Serial.println("H7 EEGNet");
  delay(2000);

  digitalWrite(LEDR,LOW);
    delay(500);
  digitalWrite(LEDR,HIGH);
    delay(500);

  digitalWrite(LEDG,LOW);
    delay(500);
  digitalWrite(LEDG,HIGH);
    delay(500);

  digitalWrite(LEDB,LOW);
    delay(500);
  digitalWrite(LEDB,HIGH);
    delay(500);  

  tflite::InitializeTarget();
    Serial.println("Initialising");
        delay(500);
  
  sdMounting_def ();

}

void loop()
{

  Serial.println("Main - NN -experimenting ");
  delay(500);

  for(int i =0; i<2; i++) //Blinking for fun 
  {
    digitalWrite(LEDG, LOW);
      delay(200);
    digitalWrite(LEDB, LOW);
      delay(200);
    digitalWrite(LEDG, HIGH);
      delay(200);
    digitalWrite(LEDB, HIGH);
      delay(200);
  }

  /*Setup Model */
  // {
    tflite::MicroProfiler profiler;
    NN_OpResolver op_resolver;
    (RegisterOps(op_resolver));

    constexpr int kTensorArenaSize = 3000;
    uint8_t tensor_arena[kTensorArenaSize];
    constexpr int kNumResourceVariables = 0;


    model = tflite::GetModel(converted_model_tflite);
        Serial.println("Model: "+ String(model->version()));
        Serial.println("Schema: "+ String(TFLITE_SCHEMA_VERSION));
          delay(500);
    tflite::MicroInterpreter interpreter_obj(
        model, op_resolver, tensor_arena, kTensorArenaSize);
      interpreter = &interpreter_obj;
      Serial.println("classic Interpreter created");
        delay(500);

      Serial.print("Tensor allocation: ");
    TfLiteStatus tensor_allo_stat = (interpreter->AllocateTensors());
      if (tensor_allo_stat == kTfLiteOk)
      {Serial.println("Success");
        delay(500);}
      else if (tensor_allo_stat != kTfLiteOk)
      {Serial.println("Fail");
        delay(500);}


    TfLiteTensor* input = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);

    Serial.println("Model created");
        delay(500);

    // basic function test
    Serial.println("Basic function test:");  
        delay(500);

    Serial.println("    input size = "+String(input->dims->size));
        delay(500);

    Serial.println("    ouput size = "+String(output->dims->size));
        delay(500);

        Serial.println(".");
    size_t used_bytes = interpreter->arena_used_bytes(); //one instance shows 872 bytes
    Serial.println("      the model occupied: "+String(used_bytes)+" bytes");
        delay(500);
        Serial.println(".");


    input->data.f[0] = 1.f;
    TfLiteStatus invoke_status = (interpreter->Invoke());
    Serial.print("    First invoke: ");
      delay(500);

    if (invoke_status !=kTfLiteOk)
    {
    Serial.println("fail.");
      delay(500);
    }
    else
    {    
    Serial.println("pass!");
      delay(500);    
    }  
    float result_t = output->data.f[0];
    Serial.println("input = 1. ; output = "+String(result_t));
  // } /*End of model setup*/

  while (true)
  {

    float number1 = random(100) / 100.0;
    float number2 = random(100) / 100.0;

    input->data.f[0] = number1;
    input->data.f[1] = number2;

    interpreter->Invoke();
    float result = output->data.f[0];

    const char *expected = number2 > number1 ? "True" : "False";
    const char *predicted = result > 0.5 ? "True" : "False";
  
    Serial.println(String(number1)+" " +String(number2) +" - result " +  String(result) + " - Expected " + String(expected) +", Predicted "+predicted);
    delay(500);
  }

}

void sdMounting_def (void)
{
  int err = fs.mount(&block_device);
  if (err)
  {
    delay(5000);
    // Serial.println("No file system found, formatting...");
    fflush(stdout);
    err = fs.reformat(&block_device);
  }
}

