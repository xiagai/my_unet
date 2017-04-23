'''
Created on Apr 22, 2017

@author: xiagai
'''
import os
import tensorflow as tf
import numpy as np
import nibabel as nib


class GetData():
    def __init__(self):
        self.data_dir = '/houseware/codalab/Liver_Tumor_Segmentation_Challenge/'
        self.n = 0
        self.i = None
        self.NUM = -1
        self.images = None
        self.labels = None
        
    def read_files(self):
        training_batch_dir = os.path.join(self.data_dir, 'Training_Batch')
        self.i = np.random.randint(0, 110)
        
        image_filename = os.path.join(training_batch_dir, 'volume-%d.nii' % self.i)
        label_filename = os.path.join(training_batch_dir, 'segmentation-%d.nii' % self.i)
        
        if not tf.gfile.Exists(image_filename):
            raise ValueError('Failed to find file: ' + image_filename)
        if not tf.gfile.Exists(label_filename):
            raise ValueError('Failed to find file: ' + label_filename)
        
        self.images = nib.load(image_filename).get_data()
        self.labels = nib.load(label_filename).get_data()
        self.NUM = self.images.shape[2]
        self.labels[self.labels == 2] = 1
    
    def read_test_files(self):
        training_batch_dir = os.path.join(self.data_dir, 'Training_Batch')
        self.i = np.random.randint(110, 131)
        
        image_filename = os.path.join(training_batch_dir, 'volume-%d.nii' % self.i)
        label_filename = os.path.join(training_batch_dir, 'segmentation-%d.nii' % self.i)
        
        if not tf.gfile.Exists(image_filename):
            raise ValueError('Failed to find file: ' + image_filename)
        if not tf.gfile.Exists(label_filename):
            raise ValueError('Failed to find file: ' + label_filename)
        
        self.images = nib.load(image_filename).get_data()
        self.labels = nib.load(label_filename).get_data()
        self.NUM = self.images.shape[2]
        self.labels[self.labels == 2] = 1
    
    def get_a_slice(self):
        if self.n + 1 >= self.NUM:
            self.read_files()
            self.n = 0
            return self.images[:, :, self.n], self.labels[:, :, self.n]
        else:
            self.n += 1
            return self.images[:, :, self.n], self.labels[:, :, self.n]

    def get_a_test_slice(self):
        if self.n + 1 >= self.NUM:
            self.read_test_files()
            self.n = 0
            return self.images[:, :, self.n], self.labels[:, :, self.n]
        else:
            self.n += 1
            return self.images[:, :, self.n], self.labels[:, :, self.n]

GetDataMachine = GetData()

def inputs():
    img, lab = GetDataMachine.get_a_slice()
    while(np.max(lab) == np.min(lab)):
        img, lab = GetDataMachine.get_a_slice()
    # img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    lab = lab.astype(np.int)
    lab2 = lab - 1
    lab2[lab2 == -1] = 1
    lab = np.reshape(lab, [512, 512, 1])
    lab2 = np.reshape(lab2, [512, 512, 1])
    lab = np.concatenate([lab, lab2], axis=2)
    return img, lab

GetTestDataMachine = GetData()

def test_inputs():
    img, lab = GetTestDataMachine.get_a_test_slice()
    while(np.max(lab) == np.min(lab)):
        img, lab = GetTestDataMachine.get_a_test_slice()
    lab = lab.astype(np.int)
    lab2 = lab - 1
    lab2[lab2 == -1] = 1
    return img, lab2

def _variable(name, shape, initializer):
    var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var

def _conv_layer(scope_name, kernal_shape, inputs):
    with tf.variable_scope(scope_name) as scope:
        kernal = _variable('weights', kernal_shape, tf.truncated_normal_initializer(stddev=1e-4, dtype=tf.float32))
        conv = tf.nn.conv2d(inputs, kernal, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable('biases', [kernal_shape[3]], tf.constant_initializer(0.01, dtype=tf.float32))
        pre_activation = tf.nn.bias_add(conv, biases)
        after_activation = tf.nn.relu(pre_activation, scope.name)
        return after_activation
    
def _upconv_layer(scope_name, kernal_shape, inputs, output_shape):
    with tf.variable_scope(scope_name) as scope:
        kernal = _variable('weights', kernal_shape, tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        up_conv = tf.nn.conv2d_transpose(inputs, kernal, output_shape, [1, 2, 2, 1], padding='SAME')
        biases = _variable('biases', [output_shape[3]], tf.constant_initializer(0.01, dtype=tf.float32))
        pre_activation = tf.nn.bias_add(up_conv, biases)
        after_activation = tf.nn.relu(pre_activation, scope.name)
        return after_activation

def _max_pool_layer(inputs, name):
    return tf.nn.max_pool(value=inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
x = tf.placeholder(tf.float32, [512, 512])
y = tf.placeholder(tf.float32, [512, 512, 2])
y_test = tf.placeholder(tf.int64, [512, 512])

image = tf.reshape(x, [1, 512, 512, 1])
label = tf.reshape(y, [1, 512, 512, 2])
label_test = tf.reshape(y_test, [512, 512])

conv1 = _conv_layer('conv1', [3, 3, 1, 32], image)
pool2 = _max_pool_layer(conv1, 'pool2') 
conv3 = _conv_layer('conv3', [3, 3, 32, 64], pool2)
pool4 = _max_pool_layer(conv3, 'pool4')
conv5 = _conv_layer('conv5', [3, 3, 64, 128], pool4)
pool6 = _max_pool_layer(conv5, 'pool6')
conv7 = _conv_layer('conv7', [3, 3, 128, 256], pool6)
pool8 = _max_pool_layer(conv7, 'pool8')
conv9 = _conv_layer('conv9', [3, 3, 256, 512], pool8)
up_conv10 = _upconv_layer('upconv10', [2, 2, 256, 512], conv9, conv7.get_shape().as_list())
concat1 = tf.concat([up_conv10, conv7], axis=3)
conv11 = _conv_layer('conv11', [3, 3, 512, 256], concat1)
up_conv12 = _upconv_layer('upconv11', [2, 2, 128, 256], conv11, conv5.get_shape().as_list())
concat2 = tf.concat([up_conv12, conv5], axis=3)
conv13 = _conv_layer('conv13', [3, 3, 256, 128], concat2)
up_conv14 = _upconv_layer('upconv14', [2, 2, 64, 128], conv13, conv3.get_shape().as_list())
concat3 = tf.concat([up_conv14, conv3], axis=3)
conv15 = _conv_layer('conv15', [3, 3, 128, 64], concat3)
up_conv16 = _upconv_layer('upconv16', [2, 2, 32, 64], conv15, conv1.get_shape().as_list())
concat4 = tf.concat([up_conv16, conv1], axis=3)
conv17 = _conv_layer('conv16', [3, 3, 64, 32], concat4)
conv18 = _conv_layer('conv18', [3, 3, 32, 2], conv17)

pos_map = tf.nn.softmax(conv18)
cross_entropy_map = label * tf.log(tf.clip_by_value(pos_map, 1e-10, 1.0))
cross_entropy = tf.reduce_sum(cross_entropy_map, axis=-1)
loss = -tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(10000):
    img, lab = inputs()
    print(i, end=', ')

    print(loss.eval(feed_dict={x: img, y: lab}, session=sess))
    train_step.run(feed_dict={x: img, y: lab}, session=sess)

prediction_map = tf.reshape(tf.argmax(pos_map, axis=3), [512, 512])
correct_prediction = tf.equal(prediction_map, label_test)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

total_accuracy = 0

for i in range(2000):
    img_test, lab_test = test_inputs()
    acc = accuracy.eval(feed_dict={x: img_test, y_test: lab_test}, session=sess)
    total_accuracy += acc
    if i % 100 == 0 and i != 0:
        print("accuracy: %.4f" % (total_accuracy / i))
