'''
Created on Apr 19, 2017

@author: xiagai
'''
import os
import tensorflow as tf
import numpy as np
import nibabel as nib

INITIAL_LEARNING_RATE = 0.01

class GetData():
    def __init__(self, isTestData):
        self.data_dir = '/houseware/codalab/Liver_Tumor_Segmentation_Challenge/'
        self.isTestData = isTestData
        self.n = 0
        self.i = None
        self.NUM = -1
        self.images = None
        self.labels = None
        
    def read_files(self):
        training_batch_dir = os.path.join(self.data_dir, 'Training_Batch')
        if self.isTestData:
            self.i = np.random.randint(110, 131)
        else:
            self.i = np.random.randint(0, 110)
        print(self.i)
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
        
    def get_a_slice(self, randomly):
        if randomly:
            if self.NUM < 0:
                self.read_files()
            self.n += 1
            if self.n > self.NUM:
                self.read_files()
                self.n = 1
            index = np.random.randint(0, self.NUM)
            return self.images[:, :, index], self.labels[:, :, index]
        else:
            if self.n + 1 >= self.NUM:
                self.read_files()
                self.n = 0
                return self.images[:, :, self.n], self.labels[:, :, self.n]
            else:
                self.n += 1
                return self.images[:, :, self.n], self.labels[:, :, self.n]

getTrainDataMachine = GetData(False)
getTestDataMachine = GetData(True)

def inputs(isTestData, useGTData, randomly):
    getDataMachine = getTestDataMachine if isTestData else getTrainDataMachine
    img, lab = getDataMachine.get_a_slice(randomly)
    if useGTData:
        while(np.max(lab) == np.min(lab)):
            img, lab = getDataMachine.get_a_slice(randomly)
        
    # img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    lab = lab.astype(np.int)
    lab2 = lab - 1
    lab2[lab2 == -1] = 1
    if isTestData:
        return img, lab2
    else:
        lab = np.reshape(lab, [512, 512, 1])
        lab2 = np.reshape(lab2, [512, 512, 1])
        lab = np.concatenate([lab, lab2], axis=2)
        return img, lab

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable.
    
    Args:
        name: Name of the variable
        shape: List of ints
        initializer: Initializer for Variable
        
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    
    Args:
        name: Name of the Variable
        shape: List of ints
        stddev: Standard deviation of a truncated Gaussion
        wd: Add L2Loss weight decay mltiplied by this float. If None, weight
            decay is not added for this Variable.
    
    Returns:
        Variable Tensor
    """
    var = _variable_on_cpu(name, shape,
                           initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _conv_layer(scope_name, kernal_shape, kernal_stddev, kernal_wd,
                inputs, biases_shape, initial_biases):    
    """Helper to create a convolution layer
    
    Args:
        scope_name: The name of variable scope
        kernal_shape: The shape of the kernal
        kernal_stddev: The stddev of kernal
        kernal_wd: The wd of kernal
        inputs: Input of the layer
        biases_shape: The shape of the biases
        initial_biases: the initial value of the biases
    
    Returns:
        Convolutional Layer
    """
    with tf.variable_scope(scope_name) as scope:
        kernal = _variable_with_weight_decay('weights', kernal_shape, kernal_stddev, kernal_wd)
        conv = tf.nn.conv2d(inputs, kernal, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', biases_shape, tf.constant_initializer(initial_biases))
        pre_activation = tf.nn.bias_add(conv, biases)
        after_activation = tf.nn.relu(pre_activation, scope.name)
        return after_activation

def _upconv_layer(scope_name, kernal_shape, kernal_stddev, kernal_wd,
                  inputs, output_shape, strides, initial_biases):
    """Helper to create a up-convolution layer
    
    Args:
        scope_name: The of the variable scope
        kernal_shape: The of the kernal
        kernal_stddev: The stddev of kernal
        kernal_wd: The wd of kernal
        inputs: Input of the layer
        strides: 1/strides is the strides of the fractional convolution
        initial_biases: The initial value of the biases
        
    Returns:
        Up-convolutional Layer
    """
    with tf.variable_scope(scope_name) as scope:
        kernal = _variable_with_weight_decay('weights', kernal_shape, kernal_stddev, kernal_wd)
        strides_shape = [1, strides, strides, 1]
        up_conv = tf.nn.conv2d_transpose(inputs, kernal, output_shape, strides_shape, 'SAME')
        biases = _variable_on_cpu('biases', output_shape[3], tf.constant_initializer(initial_biases))
        pre_activation = tf.nn.bias_add(up_conv, biases)
        after_activation = tf.nn.relu(pre_activation, scope.name)
        return after_activation
    
def _copy_and_concat(inputs, copy):
    """Concatenate the inputs and copy.
    
    Args:
        inputs: The feature maps to be cropped
        copy: The feature maps to be combined with
        
    Returns:
        The feature maps that have been combined
    """
    return tf.concat([inputs, copy], 3)
    
def inference(image):
    """Build the U-net
    
    Args:
        image: The image returned from inputs()
    
    Returns:
        The last convolution result
    """
    # conv1
    conv1 = _conv_layer(scope_name='conv1',
                        kernal_shape=[3, 3, 1, 32],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=image,
                        biases_shape=[32],
                        initial_biases=0.0)
    
    # conv2
    conv2 = _conv_layer(scope_name='conv2',
                        kernal_shape=[3, 3, 32, 32],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=conv1,
                        biases_shape=[32],
                        initial_biases=0.0)

    # pool3
    pool3 = tf.nn.max_pool(value=conv2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool3')
    
    # conv4
    conv4 = _conv_layer(scope_name='conv4',
                        kernal_shape=[3, 3, 32, 64],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=pool3,
                        biases_shape=[64],
                        initial_biases=0.0)
    
    # conv5
    conv5 = _conv_layer(scope_name='conv5',
                        kernal_shape=[3, 3, 64, 64],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=conv4,
                        biases_shape=[64],
                        initial_biases=0.0)
    
    # pool6
    pool6 = tf.nn.max_pool(value=conv5,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool6')
    
    # conv7
    conv7 = _conv_layer(scope_name='conv7',
                        kernal_shape=[3, 3, 64, 128],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=pool6,
                        biases_shape=[128],
                        initial_biases=0.0)
    
    # conv8
    conv8 = _conv_layer(scope_name='conv8',
                        kernal_shape=[3, 3, 128, 128],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=conv7,
                        biases_shape=[128],
                        initial_biases=0.0)
    
    # pool9
    pool9 = tf.nn.max_pool(value=conv8,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME',
                          name='pool9')

    # conv10
    conv10 = _conv_layer(scope_name='conv10',
                        kernal_shape=[3, 3, 128, 256],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=pool9,
                        biases_shape=[256],
                        initial_biases=0.0)
    
    # conv11
    conv11 = _conv_layer(scope_name='conv11',
                        kernal_shape=[3, 3, 256, 256],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=conv10,
                        biases_shape=[256],
                        initial_biases=0.0)
    
    # pool12
    pool12 = tf.nn.max_pool(value=conv11,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool12')
    
    # conv13
    conv13 = _conv_layer(scope_name='conv13',
                        kernal_shape=[3, 3, 256, 512],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=pool12,
                        biases_shape=[512],
                        initial_biases=0.0)
    
    # conv14
    conv14 = _conv_layer(scope_name='conv14',
                        kernal_shape=[3, 3, 512, 512],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=conv13,
                        biases_shape=[512],
                        initial_biases=0.0)
    
    # up_conv15
    up_conv15 = _upconv_layer(scope_name='up-conv15',
                              kernal_shape=[2, 2, 256, 512],
                              kernal_stddev=5e-2, 
                              kernal_wd=None,
                              inputs=conv14,
                              output_shape=conv11.get_shape().as_list(),
                              strides=2,
                              initial_biases=0.0)
    
    # copy and concatenate
    copy_and_concat1 = _copy_and_concat(up_conv15, conv11)
    
    # conv16
    conv16 = _conv_layer(scope_name='conv16',
                        kernal_shape=[3, 3, 512, 256],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=copy_and_concat1,
                        biases_shape=[256],
                        initial_biases=0.0)
    
    # conv17
    conv17 = _conv_layer(scope_name='conv17',
                        kernal_shape=[3, 3, 256, 256],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=conv16,
                        biases_shape=[256],
                        initial_biases=0.0)
    
    # up_conv18
    up_conv18 = _upconv_layer(scope_name='up-conv18',
                              kernal_shape=[2, 2, 128, 256],
                              kernal_stddev=5e-2,
                              kernal_wd=None,
                              inputs=conv17,
                              output_shape=conv8.get_shape().as_list(),
                              strides=2,
                              initial_biases=0.0)
    
    # copy and concatenate
    copy_and_concat2 = _copy_and_concat(up_conv18, conv8)
    
    # conv19
    conv19 = _conv_layer(scope_name='conv19',
                        kernal_shape=[3, 3, 256, 128],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=copy_and_concat2,
                        biases_shape=[128],
                        initial_biases=0.0)
    
    # conv20
    conv20 = _conv_layer(scope_name='conv20',
                        kernal_shape=[3, 3, 128, 128],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=conv19,
                        biases_shape=[128],
                        initial_biases=0.0)
    
    # up_conv21
    up_conv21 = _upconv_layer(scope_name='up-conv21',
                              kernal_shape=[2, 2, 64, 128],
                              kernal_stddev=5e-2,
                              kernal_wd=None,
                              inputs=conv20,
                              output_shape=conv5.get_shape().as_list(),
                              strides=2,
                              initial_biases=0.0)
    
    # copy and concatenate
    copy_and_concat3 = _copy_and_concat(up_conv21, conv5)
    
    # conv22
    conv22 = _conv_layer(scope_name='conv22',
                        kernal_shape=[3, 3, 128, 64],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=copy_and_concat3,
                        biases_shape=[64],
                        initial_biases=0.0)
    
    # conv23
    conv23 = _conv_layer(scope_name='conv23',
                        kernal_shape=[3, 3, 64, 64],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=conv22,
                        biases_shape=[64],
                        initial_biases=0.0)
    
    # up_conv24
    up_conv24 = _upconv_layer(scope_name='up-conv24',
                              kernal_shape=[2, 2, 32, 64],
                              kernal_stddev=5e-2,
                              kernal_wd=None,
                              inputs=conv23,
                              output_shape=conv2.get_shape().as_list(),
                              strides=2,
                              initial_biases=0.0)
    
    # copy and concatenate
    copy_and_concat4 = _copy_and_concat(up_conv24, conv2)
    
    # conv25
    conv25 = _conv_layer(scope_name='conv25',
                        kernal_shape=[3, 3, 64, 32],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=copy_and_concat4,
                        biases_shape=[32],
                        initial_biases=0.0)
    
    # conv26
    conv26 = _conv_layer(scope_name='conv26',
                        kernal_shape=[3, 3, 32, 32],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=conv25,
                        biases_shape=[32],
                        initial_biases=0.0)
    
    # conv27
    conv27 = _conv_layer(scope_name='conv27',
                        kernal_shape=[3, 3, 32, 2],
                        kernal_stddev=5e-2,
                        kernal_wd=None,
                        inputs=conv26,
                        biases_shape=[2],
                        initial_biases=0.0)
    
    return conv27

def loss(logits, label):
    """Calculate the losses
    
    Args:
        logits: The returns of the inference
        label: The label map of the instance
        
    Returns:
        The losses of this instance
    """
    
    pos_map = tf.nn.softmax(logits)
    cross_entropy_map = label * tf.log(tf.clip_by_value(pos_map, 1e-10, 1.0))
    cross_entropy = tf.reduce_sum(cross_entropy_map, axis=-1)
    return -tf.reduce_mean(cross_entropy)
    

def train(loss, global_step):
    """Train the model.
    
    Create an optimizer and apply to all trainable variables.
    
    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps processed.
        
    Returns:
        train_op: op for training.
    """
    opt = tf.train.GradientDescentOptimizer(INITIAL_LEARNING_RATE)
    train_op = opt.minimize(loss, global_step)
    return train_op
    