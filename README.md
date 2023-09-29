# Automatic Test Suite Reduction for Deep Neural Networks

To use DeepSuite, you need to add our callback as a subclass in your keras.callbacks.py file.

The core principle of our callback is to get a view of the internal states and statistics of the model during training.

Then you can pass our callback `DeepSuite()` to the `.fit()` method of a model as follows:

```python
callback = keras.callbacks.DeepSuite(inputs, outputs, layer_number, batch_size, startTime)
model = keras.models.Sequential()
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation(activations.relu))
model.compile(keras.optimizers.SGD(), loss='mse')
model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10, batch_size=1, 
...                     callbacks=[callback], verbose=0)
```

## Prerequisites

The version numbers below were confirmed to work with the current release:

    python 3.6.5
    Keras  2.2.0
    Keras-Applications  1.0.2
    Keras-Preprocessing 1.0.1  
    numpy 1.19.2
    pandas 1.1.5
    scikit-learn 0.21.2
    scipy 1.6.0
    tensorflow 1.14.0

    
## This repository contains the reproducibility package of DeepSuite
#### [Kaggle Models](https://github.com/ICSE2024paper/Test-Suite/tree/main/Kaggle%20Model): 
* Contains the source code of all Kaggle Models
#### [AUTOTRAINER Models]():
* Contains the source code of all AUTOTRAINER Models
#### [Spectrum Analysis]():
* Contains the source code of Callback
#### [Minimized Test Suite](https://github.com/FSE2024paper/Test-Suite/tree/main/Minimized%20Test%20Suite):
* Contains the source code of the Minimized Test Suite Algorithm 






