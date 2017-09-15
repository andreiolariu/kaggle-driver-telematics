NOTE: I no longer have the data.

This is my code for the AXA Driver Telematics challenge on Kaggle [1]. A high level description of my approach is available on my blog [2].

Code overview:
- bow.py - segments the trips using the RDP algorithm
- data_access.py - wrapper for reading trip data
- ensemble.py - code for training the ensemble, performing local testing and generating a submission:
	- do local testing: python ensemble.py weights 1 s
	- generate submission: python ensemble.py submit 1
- heading.py - segment a trip using the heading approach
- main.py - run an individual model for local testing
- model_def.py - library with all the ML algorithms used
- model_run.py - contains all the functions that fetch the data (using data_access.py), preprocess it (using util.py), apply a ML algorithm (using model_def.py) and return results
- rdp_alg.py - the RDP algorithm
- settings.py
- util.py - preprocessing data, feature extraction
- weights.py - contains all (or most of) the models tested (MODELS), as well as the final ensemble (STACK) and the weights needed for blending

The code is for educational purposes only, so expect to get your hands dirty if you play with it.

[1] https://www.kaggle.com/c/axa-driver-telematics-analysis
[2] http://webmining.olariu.org/kaggle-driver-telematics/
