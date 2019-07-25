# import numpy as np # We'll be storing our data as numpy arrays
# # import os # For handling directories
# # from PIL import Image # For handling the images
# #
# # from keras.utils import to_categorical
# # from keras import layers
# # from keras import models
# # import os.path as path
# # from sklearn.model_selection import train_test_split
# # import tensorflow as tf
# # from tensorflow.python.tools import  freeze_graph
# # from tensorflow.python.tools import optimize_for_inference_lib
# #
# # sess = tf.compat.v1.Session()
# # from keras import backend as K
# # K.set_session(sess)
# # tf.compat.v1.disable_eager_execution()
# # model_version = "2"
# # lookup = dict()
# # reverselookup = dict()
# # count = 0
# # MODEL_NAME = 'Keras_Tensorflow'
# # BATCH_SIZE = 16
# # NUM_STEPS = 3000
# #
# #
# # def model_input(input_node_name):
# #     x = tf.compat.v1.placeholder(tf.float32, shape=(1024, 1024) , name = input_node_name)
# #     y_ = tf.matmul(x, x)
# #
# #     return x, y_
# #
# #
# # def extraction_and_training():
# #     global count
# #     for j in os.listdir('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/input/leapGestRecog/00'):
# #         if not j.startswith('.'):  # If running this code locally, this is to
# #             # ensure you aren't reading in hidden folders
# #             lookup[j] = count
# #             reverselookup[count] = j
# #             count = count + 1
# #     x_data = []
# #     y_data = []
# #     datacount = 0  # We'll use this to tally how many images are in our dataset
# #     for i in range(0, 10):  # Loop over the ten top-level folders
# #         for j in os.listdir('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/input/leapGestRecog/0' + str(i) + '/'):
# #             if not j.startswith('.'):  # Again avoid hidden folders
# #                 count = 0  # To tally images of a given gesture
# #                 for k in os.listdir('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/input/leapGestRecog/0' + str(
# #                         i) + '/' + j + '/'):  # Loop over the images
# #                     # print(k)
# #                     img = Image.open('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/input/leapGestRecog/0' + str(
# #                         i) + '/' + j + '/' + k).convert('L')  # Read in and convert to greyscale
# #                     img = img.resize((320, 120))
# #                     arr = np.array(img)
# #                     x_data.append(arr)
# #                     count = count + 1
# #                     # print("Count : " , count)
# #
# #                 y_values = np.full((count, 1), lookup[j])
# #                 y_data.append(y_values)
# #                 datacount = datacount + count
# #             # print("datacount : " ,datacount)
# #     x_data = np.array(x_data, dtype='float32')
# #     y_data = np.array(y_data)
# #     y_data = y_data.reshape(datacount, 1)  # Reshape to be the correct size
# #     y_data = to_categorical(y_data)
# #     x_data = x_data.reshape((datacount, 120, 320, 1))
# #     x_data /= 255
# #     x_train, x_further, y_train, y_further = train_test_split(x_data, y_data, test_size=0.2)
# #     x_validate, x_test, y_validate, y_test = train_test_split(x_further, y_further, test_size=0.5)
# #     build_model(x_train, y_train,x_validate,y_validate,x_test,y_test)
# #     return x_validate, x_test, y_validate, y_test;
# #
# #
# # def build_model(x_train, y_train,x_validate,y_validate,x_test,y_test):
# #
# #     model = models.Sequential()
# #     model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(120, 320,1)))
# #     model.add(layers.MaxPooling2D((2, 2)))
# #     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# #     model.add(layers.MaxPooling2D((2, 2)))
# #
# #     model.add(layers.Flatten())
# #     model.add(layers.Dense(128, activation='relu'))
# #     model.add(layers.Dense(10, activation='softmax'))
# #     model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# #     model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(x_validate, y_validate))
# #     # loss and accuracy
# #     [loss, accuracy] = model.evaluate(x_test,y_test,verbose=1)
# #     print("Done")
# #
# #     merged_summary_op = model.summary()
# #     print(model.outputs)
# #     print(model.inputs)
# #
# #     return  merged_summary_op
# #
# #
# # def export_model(model_name , input_node_names, output_node_name,x,y):
# #
# #         tf.train.write_graph(K.get_session().graph_def, '1', \
# #                              model_name + '_graph.pbtxt')
# #
# #         tf.train.Saver().save(K.get_session(), '1/' + model_name + '.chkp')
# #
# #         freeze_graph.freeze_graph('1/' + model_name + '_graph.pbtxt', None, \
# #                                   False, '1/' + model_name + '.chkp', output_node_name, \
# #                                   "save/restore_all", "save/Const:0", \
# #                                   '1/frozen_' + model_name + '.pb', True, "")
# #
# #         input_graph_def = tf.GraphDef()
# #         with tf.gfile.Open('1/frozen_' + model_name + '.pb', "rb") as f:
# #             input_graph_def.ParseFromString(f.read())
# #             # import graph_def
# #
# #         output_graph_def = optimize_for_inference_lib.optimize_for_inference(
# #             input_graph_def, [input_node_names], [output_node_name],
# #             tf.float32.as_datatype_enum)
# #
# #         with tf.Graph().as_default() as graph:
# #             tf.import_graph_def(input_graph_def)
# #
# #             # print operations
# #         for op in graph.get_operations():
# #             print(op.name)
# #
# #         with tf.gfile.GFile('1/tensorflow_' + model_name + '.pb', "wb") as f:
# #             f.write(output_graph_def.SerializeToString())
# #         print("Export is done:")
# #
# #
# # def main():
# #     if not path.exists('1'):
# #         os.mkdir('1')
# #     model_name = "keras_tensorflow"
# #     input_node_name = "conv2d_1_input_1"
# #     output_node_name = "dense_2/Softmax"
# #     model_input(input_node_name)
# #     x_validate, x_test, y_validate, y_test =extraction_and_training()
# #
# #     export_model(model_name,input_node_name, output_node_name,x_validate,y_validate)
# #     print("All done!!")
# #
# #
# # if __name__ == '__main__':
# #     main()


import numpy as np # We'll be storing our data as numpy arrays
import os # For handling directories
from PIL import Image # For handling the images

from keras.utils import to_categorical
from keras import layers
from keras import models

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.tools import  freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

sess = tf.compat.v1.Session()
from keras import backend as K
K.set_session(sess)

tf.compat.v1.disable_eager_execution()
model_version = "2"
lookup = dict()
reverselookup = dict()
count = 0
MODEL_NAME = 'keras'
for j in os.listdir('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/input/leapGestRecog/00'):
    if not j.startswith('.'): # If running this code locally, this is to
                              # ensure you aren't reading in hidden folders
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1
x_data = []
y_data = []
datacount = 0 # We'll use this to tally how many images are in our dataset
for i in range(0, 10): # Loop over the ten top-level folders
    for j in os.listdir('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/input/leapGestRecog/0' + str(i) + '/'):
        if not j.startswith('.'): # Again avoid hidden folders
            count = 0 # To tally images of a given gesture
            for k in os.listdir('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/input/leapGestRecog/0' + str(i) + '/' + j + '/'): # Loop over the images
               # print(k)
                img = Image.open('C:/Users/Pepper/PycharmProjects/kerasToTensorflow/input/leapGestRecog/0' + str(i) + '/' + j + '/' + k).convert('L') # Read in and convert to greyscale
                img = img.resize((320, 120))
                arr = np.array(img)
                x_data.append(arr)
                count = count + 1
                #print("Count : " , count)

            y_values = np.full((count, 1), lookup[j])
            y_data.append(y_values)
            datacount = datacount + count
           # print("datacount : " ,datacount)
x_data = np.array(x_data, dtype = 'float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1) # Reshape to be the correct size

y_data = to_categorical(y_data)
x_data = x_data.reshape((datacount, 120, 320, 1))
x_data /= 255
x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.2)
x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)

model=models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(120, 320,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(x_validate, y_validate))
#
[loss, acc] = model.evaluate(x_test,y_test,verbose=1)
#
model.save("C:/Users/Pepper/PycharmProjects/kerasToTensorflow/input/model_tensorflow_lite.h5")


print("Done")

model.summary()
print(model.outputs)
print(model.inputs)


def export_model_for_mobile(model_name, input_node_name, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, '1', \
                         model_name + '_graph.pbtxt')

    tf.train.Saver().save(K.get_session(), '1/' + model_name + '.chkp')

    freeze_graph.freeze_graph('1/' + model_name + '_graph.pbtxt', None, \
                              False, '1/' + model_name + '.chkp', output_node_name, \
                              "save/restore_all", "save/Const:0", \
                              '1/frozen_' + model_name + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('1/frozen_' + model_name + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, [input_node_name], [output_node_name],
        tf.float32.as_datatype_enum)

    with tf.gfile.GFile('1/tensorflow_lite_' + model_name + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())


export_model_for_mobile('keras_tensor_lite', "conv2d_1_input", "dense_2/Softmax")

print("Done2")

