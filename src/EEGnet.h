#ifndef __EEGnet__
#define __EEGnet__

#include <stdint.h>
// #include <SDRAM.h>
// #include "main_functions.h" //name-mangling

namespace tflite
{
    template <unsigned int tOpCount>
    class MicroMutableOpResolver;
    class ErrorReporter;
    class Model;
    class MicroInterpreter;
} // namespace tflite

struct TfLiteTensor;

class EEGnet
{
    public:
        EEGnet();
        EEGnet(uint8_t* SDRAMptr, int RAMsize);
        float *getInputBuffer();
        void *getInputBufferAr(int index, float data);
        float predict1();
        float predict2();
        float predict3();
        float predict4();
        TfLiteTensor *input;
        TfLiteTensor *output;
        void Invoke();
        // const int kArenaSize;

        
    private:
        tflite::MicroMutableOpResolver<8> *resolver;
        tflite::ErrorReporter *error_reporter;
        const tflite::Model *model;
        tflite::MicroInterpreter *interpreter;
        
        // TfLiteTensor *input;
        // TfLiteTensor *output;
        
        uint8_t *tensor_arena;
        


};

#endif