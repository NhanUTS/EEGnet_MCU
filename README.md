# EEGnet_MCU
An implementation of the EEGnet model on a microcontroller/embedded device.  
This is done in an effort to miniaturise and move toward mobile/wearable BCi. 

# Background 
Brain computer interface, or BCi, works by translating the brain activity of the user into computer command. After the pre-processing stage is the classification of the signal. Due to the complex nature of brain signal, machine learning is strongly favoured for classification. 

EEGNet is one of the most well-known models for classification, but to the best of my awareness, so far it is only done on Desktop environments. While Desktop is good for complex computations during development, I believe embedded has the edge for actual usage mainly due to the small size and power saving. 

At the same time, it seems that running machine learning on a microcontroller is a rising need, so much so that the specific term "TinyML" has been given for the topic. Many platforms are under development to do just that, a few notable includes: Google's TensorFlow, EdgeImpulse, TexasInstrument, etc. Seeing that the tools are all here, and with support from my supervisor, we device to give it a try.  

# Description
A quick **disclamer** is I'm still an absolute novice to most things that is machine learning, programming and neuroscience. So please bear in mind that a fair amount of modifications have been done to the model and the library without any regard to the accuracy, performance and whatever else. The foremost goal of the project is to get it working. 

## Specs & Performance
At the time being, the model in use is so called 'butchered' due to having 2 layers removed compare to the original. 

The result is a drop in accuracy to 88%, comparing to 93% mentioned in the original paper.  
Memory usage is 322kB, and inference speed is 1.12s on average. 

Input shape is 60 by 151, or from the EEG perspective, that is 60-channel with 151hz sampling rate.  
Output shape is a group of 4 confidence scores, each representing a label.  
A total of 288 Epoch is used. First half for training, later half for verification.  


# Material & Resources
## Hardware
[Arduino Portenta H7](https://docs.arduino.cc/hardware/portenta-h7).  
IDE: vscode + platformio extension. 

The portenta is quite a capable arduino board with dual core, plenty of i/o function, and a large amount of RAM (1MB + 8MB). This project only utilises the M7 core, and the external 8MB is dedicated for the model. 

This project also make use of the board's SDcard function. For verification, the input epochs are preloaded into a microSD card, each epoch is reshaped into a vector, then saved into individual txt file. During inference, a single epoch (txt) is loaded into the input tensor, then invoke is called, and the result will be saved into a variable. For each epoch, its invoke time and result will be send to Serial.print and written into the result txt file, which is also saved onto the microSD.

I choose vscode+platformio mainly due to my personal preference.

Tensorflow Micro do have a [dedicated library package](https://github.com/tensorflow/tflite-micro-arduino-examples) for the ArduinoIDE. I did give it a try but my code cannot be compiled, at which I decided to move on.  

## Tensorflow for microcontroller or TFLM
[Git](https://github.com/tensorflow/tflite-micro),
[Documentation](https://www.tensorflow.org/lite/microcontrollers)   
environment: linux for building the library.   

The choice of Tensorflow is no random. Initially, Tensorflow appears to provide a near complete workflow for running machine learning on a microcontroller. You can develop your own model on GoogleColab and train it yourself, once you are happy with the performance, you can then convert it into a tflite format and finally, into a c array to run on microcontroller.

Also, arduino seems to have strong support for tensorflow and a quick google reveal that there are already many tutorials and projects available. 

The portenta is technically not listed under supported boards, but this project proven that it can use TFLM nonetheless.  

The library can be built for specific target (ie. architecture and processor). For this instance, I purposely build it with the target of "arduino-generic" since that is the framework of this whole project under platformio. 

## EEGNet
[Git](https://github.com/vlawhern/arl-eegmodels/tree/master)  
Environment: GoogleColab

EEGnet is popular, powerful and have everything about it, from the code to the training data, open source.

There are several different versions of EEGnet out there. The original github includes several variants made for specific paradigms, the one used in this project is based on the ERP.   

As mentioned earlier, TFLM doesn't have all the required operators for EEGnet. The two missing operators are Transpose and AveragePooling2D.  

The original git repo includes a trained model, which can be converted into tflite and then c array, but attempt to create an interpreter would result in a crash. 

## Quirks
During the process, I encounter a few major hiccups, this section is sort of a reflection of the whole experience, hoping that it might be useful for others. 

### TFLM itself
The git repo is still receiving new update but the web documentation is not keeping up, so keep that in mind. If you follow the documentation and encounter an error, don't panic, just go into the source file, see how it is implemented, and edit your code accordingly.

*Tips*: When search, include "TFLM" or "micro" or "microcontroller" instead of "lite", "lite" returns alot of results for mobile device for me. 

### Conflict between Arduino and TFLM
A peculiar hiccup. In vscode+platformio, when compile the script, it will tell you that there is a conflict at line 63 in the Arduino.h, the line is just defining the absolute value `abs()` function. Per the internet, this conflict is not completely unheard of, but I'm failed to find a proper fix. Since it seems relatively harmless, I decided to comment out the line and turn a blind eye, the whole thing compiles and runs fine ever since. The problem might even be platformio, but as of now, I don't have enough expertise to pin point the cause.  

# Afterthoughts 
Well, this project is obviously not perfect, there're a few things can be improved on: 

- EEGnet in its fullest*: fingercrossing while waiting for TFLM to support the two remaining operators. Alternatively, find a way to implement it myself.  

- *Optimisation*: the TFLM library can be built specifically for the M7 core, would that improve the inference speed further?

- *The Arduino framework*: per the weird conflict, would moving away from the arduino framework mitigate it? The portenta does have support for mbed framework, so that's one option.


Still, this project was eye-opening. Before this, I didn't think it was possible to run machine learning without an OS, let alone on a device with a size of a nametag. 

# Some helpful resource
[TinyML book](https://tinymlbook.com/).   
[Project: A simple neural network on ESP32](https://github.com/atomic14/tensorflow-lite-esp32).  
[Project: A handwriting recognition using Sony Spresence](https://www.hackster.io/taroyoshino007/get-started-with-tensorflow-lite-micro-by-sony-spresense-e92bf1#code). 

And if you still havent found your answer, the internet is there.  

Cheers.
