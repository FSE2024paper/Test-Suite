import keras.backend as K
from keras.callbacks import Callback
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

def indexValue(arr):
    listOfindex = []
    for index in arr:
        indices = np.where(index == 1)[0]
        listOfindex.append(indices[0])
    return listOfindex

def matchValue(match, ListIndex, epoch):
#    TestCounter = 0
    with open('label_result_{0}.txt'.format(epoch), 'a') as flabel:
         for itemNumber, itemBool in zip(ListIndex, match):
             flabel.write("{0},{1}".format( itemNumber, itemBool[itemNumber]))
             flabel.write("\n")
#             TestCounter = TestCounter + 1    
    flabel.close()


def printArrayA(my_array):
    array_string = ','.join(str(1 if elem >= 0.2 else 0) for elem in my_array)
    return array_string

    
def printArray(my_array):
    
    # Convert the array to a string with the same format
    array_string = ','.join(str(elem) for elem in my_array)
    return array_string


class OutputCallback(Callback):
    def __init__(self, X_train, y_value, layer_index=-1):
        self.X_train = X_train
        self.y_value = y_value
        self.layer_index = layer_index

    def on_epoch_end(self, epoch, logs=None):
        
        #----------------------------------------Layer 0
        self.layer_index = 2
        # Define the Keras function to get the outputs of the specified layer
        get_layer_output = K.function([self.model.input], [self.model.layers[self.layer_index].output])

        # Get the output of the specified layer for each input in X_train
        layer_outputs = get_layer_output([self.X_train])[0]

        # Print the output of the specified layer for each input in X_train
        for i, output in enumerate(layer_outputs):
            print('Output of layer {}:'.format(self.layer_index))
            print(output)
            print('')
            # Open a file in write mode
            with open('layer_output_{0}.txt'.format(self.layer_index), 'a') as fOutput:
                # Write the string to the file
                fOutput.write("T{0},{1}".format(i,printArrayA(output)))
                fOutput.write("\n")

        #----------------------------------------Layer 1
        self.layer_index = 3
        # Define the Keras function to get the outputs of the specified layer
        get_layer_output = K.function([self.model.input], [self.model.layers[self.layer_index].output])

        # Get the output of the specified layer for each input in X_train
        layer_outputs = get_layer_output([self.X_train])[0]

        # Print the output of the specified layer for each input in X_train
        for i, output in enumerate(layer_outputs):
            print('Output of layer {}:'.format(self.layer_index))
            print(output)
            print('')
            # Open a file in write mode
            with open('layer_output_{0}.txt'.format(self.layer_index), 'a') as fOutput:
                # Write the string to the file
                fOutput.write("T{0},{1}".format(i,printArrayA(output)))
                fOutput.write("\n")

        #----------------------------------------Layer 2
        self.layer_index = 4
        # Define the Keras function to get the outputs of the specified layer
        get_layer_output = K.function([self.model.input], [self.model.layers[self.layer_index].output])

        # Get the output of the specified layer for each input in X_train
        layer_outputs = get_layer_output([self.X_train])[0]

        # Print the output of the specified layer for each input in X_train
        for i, output in enumerate(layer_outputs):
            print('Output of layer {}:'.format(self.layer_index))
            print(output)
            print('')
            # Open a file in write mode
            with open('layer_output_{0}.txt'.format(self.layer_index), 'a') as fOutput:
                # Write the string to the file
                fOutput.write("T{0},{1}".format(i,printArrayA(output)))
                fOutput.write("\n")
        # Compare the predicted labels with the true labels
        pred = np.round(self.model.predict(self.X_train))
        ListIndex = indexValue(self.y_value)
        match = (pred == self.y_value)
        matchValue(match, ListIndex,epoch)
        with open('predict_result_{0}.txt'.format(epoch), 'a') as fpredict:
                # Write the string to the file
                for i, match_result in enumerate(match):
                    fpredict.write("{0}_{1}".format(i,printArray(match_result)))
                    fpredict.write("\n")
        print('Classification match:', match)


#def load_data():
#    (x_train, y_train), (x_test, y_test) = mnist.load_data()
#    x_train = x_train.astype('float32')
#    x_test = x_test.astype('float32')
#    x_train /= 255
#    x_test /= 255
#    y_train = to_categorical(y_train, num_classes=10)
#    y_test = to_categorical(y_test, num_classes=10)
#    return x_train, y_train, x_test, y_test


#def run():
#    x_train, y_train, x_test, y_test = load_data()
#model=Sequential()
#model.add(Flatten())
#input_shape=(28, 28)
#inputs = Input(shape=(input_shape))
#flatten = Flatten()(inputs)
#layer1 =Dense(5, name='dense1', activation='relu')(flatten)
##    layer1.trainable=False
##model.add(layer1)
#layer2 = Dense(7, name='dense2', activation='relu')(layer1)
##    layer2.trainable=False
##model.add(layer2)
#layer3 = Dense(9, name='dense3', activation='relu')(layer2)
##    layer3.trainable=False
##model.add(layer3)
##model.add()
#outputs = Dense(10, name='dense4', activation='softmax')(layer3)
#model = Model(inputs=inputs, outputs=outputs)
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(300,  activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

output_callback = OutputCallback(x_train, y_train)
model.compile(optimizer='Adam',metrics=['accuracy'],loss='categorical_crossentropy')
print(model.summary())
model.fit(x_train, y_train, epochs=3, verbose=1, callbacks=[output_callback])
print(model.evaluate(x_test, y_test))
#    return model





#if __name__ == '__main__':
#    et = time.time()
#    model = run()
#    elapsed_time = time.time() - et
#    print('Execution time:', elapsed_time, 'seconds')
##    freeze(model)