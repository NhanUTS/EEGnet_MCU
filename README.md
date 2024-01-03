# EEGnet_MCU
An implementation of the EEGnet model on a microcontroller/embedded device.  
This is done in an effort to miniaturise and move toward mobile/wearble BCi. 

# Background 
Brain computer interface, or BCi, works by translating the brain activity of the user into computer command. After the pre-processing stage is the classification of the signal. Due to the complex nature of brain signal, machine learning is strongly favoured for classification. 

EEGNet is one of the most well-known models for classification, but to the best of my awareness, so far it is only done on Desktop environments (The same could be said for many BCi projects in general). While Desktop is good for complex computations during development, I believe embedded has the edge for actual usage mainly due to the small size and power saving. 

At the same time, it seems that running machine learning on a microcontroller is a rising need. Many platforms is under development to do just that, a few notable includes: Google's TensorFlow, EdgeImpulse, TexasInstrument, etc. Seeing that the tools are all here, and with support from my supervisor, we device to give it a try.  

# Description
A quick **disclamer** is I'm still an absolute novice to most things that is machine learning, programming and neuroscience. So please bear in mind that a fair amount of modifications have been done to the model and the library without any regard to the accuracy, performance and whatever else. The foremost goal of the project is to get it working. 

## Specs & Performance
At the time being, the model in use is so called 'butchered' because 2 layers removed compare to the original. 

The result is a drop in accuracy to 88%, comparing to 93% mentioned in the original paper.  
Memory usage is 322kB, and inference speed is 1.12s on average. 

Input shape is 60 by 151, or from the EEG perspective is 60-channel with 151hz sampling rate.  
Output shape is a group of 4 confidence scores, each representing a label.  
A total of 288 Epoch is used. First half for training, later half for verification.  

The hardware implementation also make use of the SDcard, the inputs is stored in the 

# Material & Resources
## Hardware
Arduino Portenta H7: link.  
IDE: vscode + platformio  
*note: ArduinoIDE does not compile. Too busy to investigate


## Tensorflow for microcontroller
Source:   
environment: linux for building the library.   
*note: the current   
## EEGNet
Source:  

## Quirks 

# Afterthoughts 

# Other projects that 
