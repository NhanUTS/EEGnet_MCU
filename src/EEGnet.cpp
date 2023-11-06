#include <Arduino.h>

#include "EEGnet.h"
#include "EEGnet_model.h"


#include "butchered_model/my_EnK_q.h"
#include "butchered_model/my_ogA_f_q.h"
#include "butchered_model/my_ogA_1_q.h"
#include "butchered_model/my_ogA_m_q.h"

// #include <SDRAM.h>
// #include "main_functions.h" //name-mangling

// old tf micro library
// #include "tensorflow/lite/micro/all_ops_resolver.h"
// #include "tensorflow/lite/micro/micro_error_reporter.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// #include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
// #include "tensorflow/lite/version.h"

#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_log.h"


//up-to-date tfmicro and tensorflow library
// #include "tensorflow\lite\micro\micro_mutable_op_resolver.h"
// #include "tfLite-lib\tflite-micro-main\tensorflow\lite\micro\tflite_bridge\micro_error_reporter.h"
// #include "tfLite-lib\tflite-micro-main\tensorflow\lite\micro\micro_interpreter.h"
// // #include "tfLite-lib\tflite-micro-main\tensorflow\lite\schema\schema_generated.h"

// #include ""
// #include ""
// #include ""



// const int kArenaSize = 30*EEGNET_tfLite_tflite_len;
// const int kArenaSize = 32*13748;
// const int kArenaSize = 445000;

EEGnet::EEGnet()
// EEGnet::EEGnet()
{
    Serial.println("internal RAM");
    const int kArenaSize = 412440;

    // error_reporter = new tflite::MicroErrorReporter();
    
    // model = tflite::GetModel(EEGNET_tfLite_tflite);

    // model = tflite::GetModel(my_EEGnet_EnK_q_tflite);
    model = tflite::GetModel(my_EEGnet_ogA_f_q_tflite);
    // model = tflite::GetModel(my_EEGnet_ogA_1_q_tflite);
    // model = tflite::GetModel(my_EEGnet_ogA_m_q_tflite);

    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        // TF_LITE_REPORT_ERROR(error_reporter, "Model provided is schema version %d not equal to supported version %d.",
        //                      model->version(), TFLITE_SCHEMA_VERSION);

        // 
        Serial.println("Model provided is schema version " + String(model->version()) 
                        + " not equal to supported version: " + String(TFLITE_SCHEMA_VERSION));
        delay(2000);
        return;
    }
    delay(1000);
    Serial.println("GetModel()- Done");


    // This pulls in the operators implementations we need
    resolver = new tflite::MicroMutableOpResolver<8>();
    delay(1000);
    Serial.println("resolver create- Done");

    // resolver->AddTranspose();
    resolver->AddConv2D();
    resolver->AddDepthwiseConv2D();
    resolver->AddElu();

    resolver->AddMul();
    resolver->AddAdd();

    resolver->AddReshape();
    resolver->AddFullyConnected();
    resolver->AddSoftmax();


    delay(1000);
    Serial.println("Adding Operators- Done");
    
    
    tensor_arena = (uint8_t *)malloc(kArenaSize);
    // uint8_t* tensor_arena;    
    // alignas(16) uint8_t tensor_arena[kArenaSize];
    // uint8_t* tensor_arena = SDRAMptr;

    delay(1000);
    Serial.println("Tensor Arena ");

    // uint8_t tensor_arena[kArenaSize];
    if (!tensor_arena)
    {
        // TF_LITE_REPORT_ERROR(error_reporter, "Could not allocate arena");

        Serial.println("Could not allocatae arena");
        delay(2000);

        return;
    }

    // Build an interpreter to run the model with.
    interpreter = new tflite::MicroInterpreter(
        model, *resolver, tensor_arena, kArenaSize);

    delay(1000);
    Serial.println("Create interpreter- Done");

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        // TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");

        Serial.println("AllocateTensor() failed");
        delay(2000);
        return;
    }
    delay(1000);
    Serial.println("allocate status- Done");


    size_t used_bytes = interpreter->arena_used_bytes();
    // TF_LITE_REPORT_ERROR(error_reporter, "Used bytes %d\n", used_bytes);

    Serial.println("Used bytes: " + String(used_bytes));
    delay(2000);

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);

}

EEGnet::EEGnet(uint8_t* SDRAMptr, int RAMsize)
// EEGnet::EEGnet()
{
    Serial.println("extra SDRAM");
    const int kArenaSize = RAMsize;
    // error_reporter = new tflite::MicroErrorReporter();

    // model = tflite::GetModel(EEGNET_tfLite_tflite);

    // model = tflite::GetModel(my_EEGnet_EnK_q_tflite);
    model = tflite::GetModel(my_EEGnet_ogA_f_q_tflite);
    // model = tflite::GetModel(my_EEGnet_ogA_1_q_tflite);
    // model = tflite::GetModel(my_EEGnet_ogA_m_q_tflite);

    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        // TF_LITE_REPORT_ERROR(error_reporter, "Model provided is schema version %d not equal to supported version %d.",
        //                      model->version(), TFLITE_SCHEMA_VERSION);

        
        Serial.println("Model provided is schema version " + String(model->version()) 
                        + " not equal to supported version: " + String(TFLITE_SCHEMA_VERSION));
        delay(2000);
        return;
    }
    delay(1000);
    Serial.println("GetModel()- Done");


    // This pulls in the operators implementations we need
    resolver = new tflite::MicroMutableOpResolver<8>();
    delay(1000);
    Serial.println("resolver create- Done");
    // resolver->AddDequantize();
    // add transpose()
    // resolver->AddTranspose();
    resolver->AddConv2D();
    resolver->AddDepthwiseConv2D();
    resolver->AddElu();

    resolver->AddMul();
    resolver->AddAdd();
    // resolver->AddElu();

    resolver->AddReshape();
    resolver->AddFullyConnected();
    resolver->AddSoftmax();

    delay(1000);
    Serial.println("Adding Operators- Done");
    
    // uint8_t tensor_arena[kArenaSize];
    // tensor_arena = (uint8_t *)malloc(kArenaSize);
    // uint8_t* tensor_arena;
    // tensor_arena = (uint8_t*)SDRAMptr->malloc(kArenaSize); //
   
    uint8_t* tensor_arena = SDRAMptr;

    delay(1000);
    Serial.print("Tensor Arena: ");

    // uint8_t tensor_arena[kArenaSize];
    if (!tensor_arena)
    {
        // TF_LITE_REPORT_ERROR(error_reporter, "Could not allocate arena");

        Serial.println("Could not allocatae arena");
        delay(2000);

        return;
    }
    else
    {
        Serial.println("success");
        delay(2000);
    }

    // Build an interpreter to run the model with.
    interpreter = new tflite::MicroInterpreter(
        model, *resolver, tensor_arena, kArenaSize);

    delay(1000);
    Serial.println("Create interpreter- Done");

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        // TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");

        Serial.println("AllocateTensor() failed");
        delay(2000);
        return;
    }
    delay(1000);
    Serial.println("allocate status- Done");


    size_t used_bytes = interpreter->arena_used_bytes();
    // TF_LITE_REPORT_ERROR(error_reporter, "Used bytes %d\n", used_bytes);

    Serial.println("Used bytes: " + String(used_bytes));
    delay(2000);

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);

}

float *EEGnet::getInputBuffer()
{
    return input->data.f;
}

void *EEGnet::getInputBufferAr(int index, float data)
{
     input->data.f[index] = data;
}


void EEGnet::Invoke()
{
    interpreter->Invoke();
}
float EEGnet::predict1()
{
    // interpreter->Invoke();
    return output->data.f[0];
}
float EEGnet::predict2()
{
    // interpreter->Invoke();
    return output->data.f[1];
}
float EEGnet::predict3()
{
    // interpreter->Invoke();
    return output->data.f[2];
}
float EEGnet::predict4()
{
    // interpreter->Invoke();
    return output->data.f[3];
}