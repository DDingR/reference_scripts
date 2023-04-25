import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets.mnist import load_data

import numpy as np
# import m

class Learner(Model):
    
    def __init__(self):
        super(Learner, self).__init__()
        
        self.h1 = Dense(128, activation='relu')
        self.h2 = Dense(64, activation='relu')
        self.h3 = Dense(10, activation='softmax')

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        # return np.argmax(x)
        return x

def learn(L, input, label):
    input = np.reshape(input, [1, 28*28])
    input = tf.convert_to_tensor(input, dtype=tf.float32)
    result = L(input)
    # result = np.argmax(result)
    
    # result = tf.convert_to_tensor(result, dtype=tf.float32)
    # label = tf.convert_to_tensor(label, dtype=tf.float32)
    label_list = np.zeros([1,10])
    label_list[0][label] = 1

    label_list = tf.convert_to_tensor(label_list, dtype=tf.float32)


    print(label_list)
    print(result)

    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(tf.losses.categorical_crossentropy(label_list, result))
    print(loss)
    grads = tape.gradient(loss, result)
    print(grads)
    opt_option.apply_gradients(zip(grads, L.trainable_variables))


(train_images, train_labels), (test_images, test_labels) = load_data()
# print(train_images[0])

L = Learner()
L.build(input_shape=(None, 28*28))
L.summary()

opt_option = Adam(0.001)


# for i in range(len(train_images)):
for i in range(1000):
    # print(L(tf.convert_to_tensor(np.reshape(train_images[i], [1,28*28]))))
    # print(np.argmax(L(tf.convert_to_tensor(np.reshape(train_images[i], [1,28*28])))))
    learn(L, train_images[i], train_labels[i])
    print(i, end='\r')

for i in range(100):
    cnt = 0
    input = test_images[i]
    label = test_labels[i]

    input = np.reshape(input, [1, 28*28])
    input = tf.convert_to_tensor(input)
    result = L(input)
    
    if np.argmax(result) == label:
        cnt = cnt + 1

print(cnt)