# Multiscale Neural Network (MNN) based on hierarchical nested bases
This repo is the code for the paper: https://arxiv.org/abs/1808.02376

The code to generate data is written by __MATLAB__ and the neural network is implemented by python based on __keras__ on top of __tensorflow__

## How to run
1. generate data by matlab code
2. run the code in the tensorflow environment 

We take NLSE 1D as an example.

1. open __MATLAB__ and run the code _NLSEsample.m_ to general data. 
   it may cost tens of minutes
2. make sure you have installed __keras__ and __tensorflow__ ([install tensorflow](https://www.tensorflow.org/install/))
	1. run
	```
		python testH2matrix.py --help  
	```
	to print the usage of all the parameters and its default values.
	
	2. run the code by setting the parameters. One example:
	```
		python testH2matrix.py --epoch 2000 --alpha 5 --output-suffix V1
	```
	One can also direct run it with default values as
	```
		python testH2matrix.py
	```


We note that the code for Kohn-Sham map is same as that for NLSE, thus we only provide the codes for NLSE and RTE.
