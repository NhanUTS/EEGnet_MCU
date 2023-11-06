#include <Arduino.h>

#include "NeuralNetwork.h"
#include "model_data.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// #include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
// #include "tensorflow/lite/version.h"

const int kArenaSize = 20000;

NeuralNetwork::NeuralNetwork()
{
    Serial.println("test model - internal ram");
    delay(1000);

    model = tflite::GetModel(converted_model_tflite);
    // if (model->version() != TFLITE_SCHEMA_VERSION)
    // {
    //     TF_LITE_REPORT_ERROR(error_reporter, "Model provided is schema version %d not equal to supported version %d.",
    //                          model->version(), TFLITE_SCHEMA_VERSION);
    //     return;
    // }
    // This pulls in the operators implementations we need  
        Serial.println("GetModel()- Done");
        delay(1000);

    resolver = new tflite::MicroMutableOpResolver<10>();
        delay(1000);
        Serial.println("resolver create- Done");

    resolver->AddFullyConnected();
    resolver->AddMul();
    resolver->AddAdd();
    resolver->AddLogistic();
    resolver->AddReshape();
    resolver->AddQuantize();
    resolver->AddDequantize();

        Serial.println("Adding Operators- Done");
        delay(1000);

    uint8_t* tensor_arena;
    tensor_arena = (uint8_t *)malloc(kArenaSize);
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
        
        Serial.println("Create interpreter- Done");
        delay(1000);

        Serial.println("pre-Allo");
        delay(1000);
    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
        Serial.println("Allo");
        delay(1000);
    if (allocate_status != kTfLiteOk)
    {
        // TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        
        Serial.println("AllocateTensor() failed");
        delay(2000);
        return;
    }
   
    Serial.println("allocate status: Done");
    delay(1000);

    size_t used_bytes = interpreter->arena_used_bytes();
    // TF_LITE_REPORT_ERROR(error_reporter, "Used bytes %d\n", used_bytes);

        Serial.println("Used bytes: " + String(used_bytes));
        delay(2000);

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);
}

NeuralNetwork::NeuralNetwork(uint8_t* SDRAMptr)
{
    // error_reporter = new tflite::MicroErrorReporter();
    Serial.println("test model");
    delay(1000);
    Serial.println("external ram");
    delay(1000);

    model = tflite::GetModel(converted_model_tflite);
    // if (model->version() != TFLITE_SCHEMA_VERSION)
    // {
    //     TF_LITE_REPORT_ERROR(error_reporter, "Model provided is schema version %d not equal to supported version %d.",
    //                          model->version(), TFLITE_SCHEMA_VERSION);
    //     return;
    // }
    // This pulls in the operators implementations we need
   
    
    Serial.println("GetModel()- Done");
    delay(1000);

    resolver = new tflite::MicroMutableOpResolver<10>();
        delay(1000);
        Serial.println("resolver create- Done");

    resolver->AddFullyConnected();
    resolver->AddMul();
    resolver->AddAdd();
    resolver->AddLogistic();
    resolver->AddReshape();
    resolver->AddQuantize();
    resolver->AddDequantize();

        Serial.println("Adding Operators- Done");
        delay(1000);

        
    // tensor_arena = (uint8_t *)malloc(kArenaSize);

    uint8_t* tensor_arena = SDRAMptr;
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
        
        Serial.println("Create interpreter- Done");
        delay(1000);

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        // TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        
        Serial.println("AllocateTensor() failed");
        delay(2000);
        return;
    }
   
    Serial.println("allocate status: Done");
    delay(1000);

    size_t used_bytes = interpreter->arena_used_bytes();
    // TF_LITE_REPORT_ERROR(error_reporter, "Used bytes %d\n", used_bytes);

        Serial.println("Used bytes: " + String(used_bytes));
        delay(2000);

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);
}

float *NeuralNetwork::getInputBuffer()
{
    return input->data.f;
}

void NeuralNetwork::invoke()
{
    interpreter->Invoke();
}

float NeuralNetwork::predict()
{
    // interpreter->Invoke();
    return output->data.f[0];
}
