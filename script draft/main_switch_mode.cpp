// #include <Arduino.h>  //comment out line 63, weird conflict redefinition of abs()
                      //might be standard/compatibility/optimisation related 

// #include <main_functions.h>
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

// Original model with full layers
// #include "EEGnet.h"
// #include "EEGnet_model.h"

#include "butchered_model/my_EEGnet_ogA_1_h5_mc.h"

// Butchered models aka removed layers that has not been supported by the official tflite-micro
#include "butchered_model/my_EnK_q.h"
#include "butchered_model/my_ogA_f_q.h"
#include "butchered_model/my_ogA_1_q.h"
#include "butchered_model/my_ogA_m_q.h"

//Simple neural network model for testing purpose
#include "test_model/NeuralNetwork.h"
#include "test_model/model_data.h"

#include <SDRAM.h>

#include "SDMMCBlockDevice.h"
#include "FATFileSystem.h"

// For SDram
#define SDRAM_ARRAY_BYTES 8000000 // give 8MB of SDRAM

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

// NeuralNetwork *nn;
bool test = false;  

namespace 
{
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  int inference_count = 0;

  uint8_t* tensor_arena = (uint8_t*)SDRAM_START_ADDRESS;


  using EEGnet_OpResolver = tflite::MicroMutableOpResolver<40>; //important!! - try put more than need

  TfLiteStatus RegisterOps(EEGnet_OpResolver& op_resolver) 
  {
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddElu());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());

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


  //mount device if needed
  sdMounting_def ();
    Serial.println("SDcard: done");
        delay(500);

  tflite::InitializeTarget();
    Serial.println("Initialising");
        delay(500);

  SDram.begin(SDRAM_START_ADDRESS+ SDRAM_ARRAY_BYTES);
    Serial.println("SDram: started");
        delay(500);

  // SDaddress = (uint8_t*)SDram.malloc(SDRAM_ARRAY_BYTES);

  // RAMptr = (uint8_t*)SDRAM_START_ADDRESS;
  // nn = new NeuralNetwork();
  // nn = new NeuralNetwork(RAMptr);
  



}

void loop()
{
  /*Setup Model */
  // {

    Serial.println("Model setup");
      delay(500);

    // tflite::MicroProfiler profiler;
    // EEGnet_OpResolver op_resolver;
    // (RegisterOps(op_resolver));

    // constexpr long int kTensorArenaSize = 2000000;
    //     Serial.println("kTnAr ="+String(kTensorArenaSize));
    //     delay(200);

    // // uint8_t tensor_arena[kTensorArenaSize];
    //     // Serial.println("TensorArena ="+String(tensor_arena));
    //     // delay(200);

    // // constexpr int kNumResourceVariables = 24; //not yet clear the use of this variable


    // model = tflite::GetModel(my_EEGnet_ogA_1_h5_tflite);
    //     Serial.println("Model: "+ String(model->version()));
    //     Serial.println("Schema: "+ String(TFLITE_SCHEMA_VERSION));
    //       delay(500);
    // tflite::MicroInterpreter interpreter_obj(
    //     model, op_resolver, tensor_arena, kTensorArenaSize);
    //   interpreter = &interpreter_obj;
    //   Serial.println("classic Interpreter created");
    //     delay(500);

    //   Serial.print("Tensor allocation: ");
    // TfLiteStatus tensor_allo_stat = (interpreter->AllocateTensors());
    //   if (tensor_allo_stat == kTfLiteOk)
    //   {Serial.println("Success");
    //     delay(500);}
    //   else if (tensor_allo_stat != kTfLiteOk)
    //   {Serial.println("Fail");
    //     delay(500);}


    // TfLiteTensor* input = interpreter->input(0);
    // TfLiteTensor* output = interpreter->output(0);

    // Serial.println("Model created");
    //     delay(500);

    //   // basic function test
    // Serial.println("Basic function test:");  
    //     delay(500);

    // Serial.println("    input size = "+String(input->dims->size));
    //     delay(500);

    // Serial.println("    ouput size = "+String(output->dims->size));
    //     delay(500);

    //     Serial.print("    .");
    // size_t used_bytes = interpreter->arena_used_bytes();
    // Serial.println("the model occupied: "+String(used_bytes)+"bytes");
    //     delay(500);
    //     Serial.print("  .");

    // input->data.f[0] = 1.f;
    
    //     float ts1, te1, dt1, st1;
    //     ts1 = micros();
    // TfLiteStatus invoke_status = (interpreter->Invoke());
    //     te1 = micros();
    // Serial.print("First invoke: ");
    //   delay(500);

    // if (invoke_status !=kTfLiteOk)
    // {
    // Serial.println("fail :( ");
    //   delay(500);
    // }
    // else
    // {    
    // Serial.print("pass!");
    //   delay(500);    
    // }  

    //     dt1 = te1-ts1;
    //     Serial.println(" : " +String(dt1)+"us");

    // float result_t = output->data.f[0];
    // Serial.println("    input = 1. ; output = "+String(result_t));
  // } /*End of model setup*/  
    
    
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

      float ts, te, dt, st;
      // // st = 1800;
      // //   Serial.println("time set: " +String(st));
      // // ts = millis();
      // //   delay(1800);
      // // te = millis();
      // // dt = te-ts;
      // // Serial.println("time measure: " +String(dt));

int mode = 2;
switch (mode)
{
  case 0:

  Serial.println("Mode:" +String(mode));
  // mode = 3;
  mode++;
    break;

  case 1:   
  Serial.println("EEGnet - Single Invoke");
  delay(500);

  // while (test == false) //running the EEG model
  // {
  //     Serial.println("- Variable setup ");
  //     delay(500);

  //     const int noE = 288;    //no. of epochs
  //     const int channel =60;  //no. of channel
  //     const int tp =151;      //sampling rate
  //     const int label = 4;    //no. of label 
  //     int dtp = channel*tp;   //total data point for a single predict

  //     float indata = 0; 
  //     float re[label] = {0, 0, 0, 0}; //label array

      
  //     // {
  //     //   Serial.print("- Model creating ");
  //     //   delay(500);
  //     //   // eegNET = new EEGnet();
  //     //   eegNET = new EEGnet(RAMptr,SDRAM_ARRAY_BYTES);

  //     //   Serial.println("It's here !!!!");
  //     // }
      
      
  //     // eegNET
  //     //   Serial.println("Model kSize = " +String(EEGNET_))
  //     // eegNET->getInputBuffer()[0] = number1;
  //     // eegNET->getInputBuffer()[1] = number2;
  //     char mes[60];  
      
  //     FILE * fr;       
  //       Serial.println("Open data file");
  //       delay(500);
  //     fr = fopen ("fs/eegNet/mne_e1_arg.txt", "r");
      

  //     while (fr==NULL)
  //     {
  //       // delay(500);
  //       // Serial.println("     file open fail!!!");
  //       // Serial.println("     retry open file\n");
  //       fr = fopen ("fs/eegNet/mne_e1_arg.txt", "r");
  //     }

  //     FILE * fw; 
  //       Serial.println("Open write file");
  //       delay(500);
  //     fw = fopen ("fs/eegNet/ftest_results.txt", "a");

  //     while (fw==NULL)
  //     {
  //       delay(500);
  //       // Serial.println("     file open fail!!!");
  //       // Serial.println("     retry open file\n");
  //       fw = fopen ("fs/eegNet/ftest_results.txt", "a");
  //     }
      
  //       Serial.println("loading data");
  //       delay(500);
  //     // while(!feof(fr))
  //       ts = millis();
  //     for(int i =0; i<dtp; i++)
  //     {
  //       fgets(mes, 60, fr); //get the data from txt file as string
  //     //   // *(data_p+index) = strtof(mes,NULL) + var;
  //       int idx = i;
  //       // Serial.print(".");
  //       indata = strtof(mes,NULL);
  //       input->data.f[i] = indata;
  //       // Serial.print(".");
  //       // fprintf(fw,"%f\n",(indata));

  //       // eegNET->getInputBuffer()[i] = indata;

  //       // (eegNET)->input->data.f[i] = indata;
  //       // eegNET->getInputBufferAr(idx, indata) ;
  //       // Serial.print(".");
  //     }
  //       te = millis();
  //       dt = te-ts;

  //     fclose(fr); //close reading file
  //       Serial.println("Data loaded: "+String(dt)+"ms");
  //       delay(500);
      
  //     // float result = eegNET->predict();
      
  //       Serial.print("Model invoke: ");
  //       delay(500);
  //     // {
  //               ts = micros();
  //           TfLiteStatus invoke_status = interpreter->Invoke();
  //               te = micros();
  //               dt = te-ts;
  //             if (invoke_status !=kTfLiteOk)
  //             {
  //               Serial.println("fail.");
  //                 delay(500);
  //             }
  //             else
  //             {    
  //               Serial.println("pass!");
  //                 delay(500);
  //             }                           
  //               Serial.println("    invoke time: " +String(dt)+"us");
        
  //       for (int i =0; i<label; i++)
  //       {
  //         re[i] = output->data.f[i];
  //       }
      
  //     Serial.println("\n.\n.\n.\n ");
  //     delay(500);
      
  //     Serial.println("Output:");
  //     delay(500);

  //     for (int i =0; i<label; i++)    //print label to serial
  //     {
  //       Serial.println("  Label "+String(i+1)+": " + String(re[i]));
  //       delay(1000); 
  //     }

  //     delay(500);
  //     Serial.println(".");

  //     for(int i =0; i<label; i++)   //write label to txt file
  //     {
  //       fprintf(fw,"Label %i: ",i);
  //       fprintf(fw,"%f\n",re[i]);
  //     }
      
  //     fclose(fw);

  //     Serial.println("chilling ... ");
  //     while(true)
  //     {
  //       sleep();
  //     }
  //   } 
  
    mode++;
    break;

  case 2:
    Serial.println("EEGnet - Continous mode");
      delay(500);

    //   char* readfile_ap="fs/data_test/tr_w";
    //   char* f_xtn = ".txt";

    // for (int i =0; i<noE; i++)
    // {
    //   FILE* fr;
    //   // char file_name = (readfile_ap+String(i)+f_xtn);
    //   // char* fn_p = &file_name ;       
    //   char file_name[10];
    //   sprintf(file_name,"%s%i%s",readfile_ap,i,f_xtn);
    //     Serial.print("Opening file: ");
    //     Serial.println((file_name));
    //     delay(500);
    //   fr = fopen (file_name, "r");

    //   // while (fr==NULL)
    //   // {
    //   //   fr = fopen ("fs/eegNet/mne_e1_arg.txt", "r");
    //   // }
    //   char mes[10];
    //   float indata;
    //   while(!feof(fr))
    //   {
    //     fgets(mes, 60, fr);
    //     indata = strtof(mes,NULL)+100;

    //     Serial.println("    "+String(indata));
    //       delay(200);
    //   }
    // }
    mode++;
    break;

  case 3:
    break;

  case 4:  
    Serial.println("consequential file read test");
      delay(500);
  // while (true)
  // {

  //   const int noE = 23;    //no. of epochs
  //   char* readfile_ap="fs/data_test/tr_w";
  //   char* f_xtn = ".txt";

  //   // char* a = "tr_w";
  //   // char* b = ".txt";
  //   // char c = c;
  //   // // char ab = a+b;
  //   // char noo[10];
  //   // sprintf(noo,"%c%c and %c", a,b,c);
  //   // Serial.println(noo);
  //   //   delay(500);


  //     // char* str_p;
  //     // str_p = &f_xtn;

  //   for (int i =0; i<noE; i++)
  //   {
  //     FILE* fr;
  //     // char file_name = (readfile_ap+String(i)+f_xtn);
  //     // char* fn_p = &file_name ;       
  //     char file_name[10];
  //     sprintf(file_name,"%s%i%s",readfile_ap,i,f_xtn);
  //       Serial.print("Opening file: ");
  //       Serial.println((file_name));
  //       delay(500);
  //     fr = fopen (file_name, "r");

  //     // while (fr==NULL)
  //     // {
  //     //   fr = fopen ("fs/eegNet/mne_e1_arg.txt", "r");
  //     // }
  //     char mes[10];
  //     float indata;
  //     while(!feof(fr))
  //     {
  //       fgets(mes, 60, fr);
  //       indata = strtof(mes,NULL)+100;

  //       Serial.println("    "+String(indata));
  //         delay(200);
  //     }
  //   }
  // }
  mode++;
    break;

  case 10: // a simple neural network, for testing purpose
    Serial.println("simple Neural Network");
      delay(500);

  
  // while (test == true)    
  // {

  //   float number1 = random(100) / 100.0;
  //   float number2 = random(100) / 100.0;

  //   input->data.f[0] = number1;
  //   input->data.f[1] = number2;

  //   interpreter->Invoke();
  //   float result = output->data.f[0];


  //   const char *expected = number2 > number1 ? "True" : "False";
  //   const char *predicted = result > 0.5 ? "True" : "False";
  
  //   Serial.println(String(number1)+" " +String(number2) +" - result " +  String(result) + " - Expected " + String(expected) +", Predicted "+predicted);
  // }
  mode = 0;
    break;  
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

