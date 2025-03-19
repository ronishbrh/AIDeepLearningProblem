# train.cpp
It contains the code for training a model. All various models were trained using this file by modifying it. Right now, it trains for the configuration of Leaky ReLU + Softmax + Cross-Entropy Loss

# train_adam.cpp
It contains the code for training a model using ADAM optimizer but is incomplete.

# run.cpp
It contains the code for running the model.

# Compiling and running
## In Linux
### For train.cpp
1. Run the following command
```BASH
g++ train.cpp; ./a.out
```
### For run.cpp
1. Install opencv
2. Run the following command
```BASH
g++ run_linux.cpp `pkg-config --cflags --libs opencv4`; ./a.out
```
## In Windows
### For train.cpp
### For run.cpp
