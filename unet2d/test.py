'''
Created on Apr 20, 2017

@author: xiagai
'''
import tensorflow as tf
from unet2d import functions


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/houseware/codalab/Liver_Tumor_Segmentation_Challenge/experiment/training_result',
                           """Directory where to write event logs"""
                           """and checkpoint.""")

def evaluate():
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [512, 512])
        # y = tf.placeholder(tf.int64, [512, 512])
        
        image = tf.reshape(x, [1, 512, 512, 1])
        # label = tf.reshape(y, [512, 512])
        
        logits = functions.inference(image, 32)
        pos_map = tf.nn.softmax(logits)
        
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            saver.restore(sess, FLAGS.train_dir + '/model.ckpt')
            functions.evaluate(pos_map, x, sess, 110, 131, useGTData=True, plot=False)

evaluate()
