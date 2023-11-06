/* a collection of previous failed attemp
      // { //arduino-example
      //   tflite::InitializeTarget();

      //   // Map the model into a usable data structure. This doesn't involve any
      //   // copying or parsing, it's a very lightweight operation.
      //   model = tflite::GetModel(converted_model_tflite);
      //   if (model->version() != TFLITE_SCHEMA_VERSION) {
      //     MicroPrintf(
      //         "Model provided is schema version %d not equal "
      //         "to supported version %d.",
      //         model->version(), TFLITE_SCHEMA_VERSION);
      //     return;
      //   }

      //   // This pulls in all the operation implementations we need.
      //   // NOLINTNEXTLINE(runtime-global-variables)
      //   static tflite::MicroMutableOpResolver<8> resolver;
      //         resolver.AddFullyConnected();
      //         resolver.AddMul();
      //         resolver.AddAdd();
      //         resolver.AddLogistic();
      //         resolver.AddReshape();
      //         resolver.AddQuantize();
      //         resolver.AddDequantize();

      //   // Build an interpreter to run the model with.
      //   static tflite::MicroInterpreter static_interpreter(
      //       model, resolver, tensor_arena, kTensorArenaSize);
      //   interpreter = &static_interpreter;

      //       delay(1000);
      //       Serial.println("Create interpreter- Done");

      //   // Allocate memory from the tensor_arena for the model's tensors.
      //   TfLiteStatus allocate_status = interpreter->AllocateTensors();
      //   if (allocate_status != kTfLiteOk) {
      //     MicroPrintf("AllocateTensors() failed");
      //     return;
      //   }
      //     delay(1000);
      //     Serial.println("allocate status- Done");
      //     size_t used_bytes = interpreter->arena_used_bytes();
      //     Serial.println("Used bytes: " + String(used_bytes));
      //     delay(2000);
      //   // Obtain pointers to the model's input and output tensors.
      //   input = interpreter->input(0);
      //   output = interpreter->output(0);

      //   // Keep track of how many inferences we have performed.
      //   inference_count = 0;
      // }
  
  {
    // Serial.print("- Model creating ");
    // delay(500);
    // eegNET = new EEGnet();
    // // eegNET = new EEGnet(RAMptr,SDRAM_ARRAY_BYTES);

    // Serial.println("It's here !!!!");
    // test = false;
  }
  
  // Serial.flush();
  // Serial.println("SD card test");

  // SDram.begin(SDRAM_START_ADDRESS+SDRAM_ARRAY_BYTES);
  // SDaddress = (uint8_t*)SDram.malloc(32*EEGNET_tfLite_tflite_len);
  // SDaddress = (uint8_t*)SDram.malloc(610*EEGNET_tfLite_tflite_len);
  // SDaddress = (uint8_t*)SDram.malloc(SDRAM_ARRAY_BYTES);



  // alloc preset amount of bytes in SDRAM
  // SDaddress = (uint8_t*)SDram.malloc(SDRAM_ARRAY_BYTES);

      // SDRAM test
  // while(true) 
  // {
  //   // fill allocated SDRAM
  //   for(int i = 0; i < SDRAM_ARRAY_BYTES; i++) {
  //     SDaddress[i] = SDRAM_FILL_VALUE;
  //   }

  //   bool correct = true; // test pass flag
  //   int counter = 0;     // counter for number of errors detected

  //   // check filled SDRAM
  //   for(int i = 0; i < SDRAM_ARRAY_BYTES; i++) {
  //     if(SDaddress[i] != SDRAM_FILL_VALUE) {
  //       // incorrect byte
  //       correct = false;
  //       counter++;
  //     }
  //   }

  //   delay(2000);
  //   // log results
  //   if(correct) 
  //   {
  //     // test passed
  //     Serial.println("SDRAM test passed");
  //   }
  //   else 
  //   {
  //     // test failed
  //     Serial.print("ERROR: SDRAM test failed with ");
  //     Serial.print(counter);
  //     Serial.println(" errors");
  //   }
  // }
*/