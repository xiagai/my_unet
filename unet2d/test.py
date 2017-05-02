'''
Created on Apr 20, 2017

@author: xiagai
'''
import tensorflow as tf
from unet2d import functions


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '//houseware/codalab/Liver_Tumor_Segmentation_Challenge/experiment/training_result3',
                           """Directory where to write event logs"""
                           """and checkpoint.""")

def evaluate(isEvaluation):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [512, 512])
        image = tf.reshape(x, [1, 512, 512, 1])        
        logits = functions.inference(image, 32)
        pos_map = functions.softmax(logits)
        
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            if isEvaluation:
                saver.restore(sess, tf.train.get_checkpoint_state(FLAGS.train_dir).model_checkpoint_path)
                functions.evaluate(pos_map, x, sess, 110, 131, useGTData=True, plot=False)
            else:
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
                path = ckpt.all_model_checkpoint_paths
                saver.restore(sess, path[0])
                functions.evaluate(pos_map, x, sess, 100, 110, useGTData=True, plot=False)
                saver.restore(sess, path[1])
                functions.evaluate(pos_map, x, sess, 100, 110, useGTData=True, plot=False)
                saver.restore(sess, path[2])
                functions.evaluate(pos_map, x, sess, 100, 110, useGTData=True, plot=False)
                saver.restore(sess, path[3])
                functions.evaluate(pos_map, x, sess, 100, 110, useGTData=True, plot=False)
                saver.restore(sess, path[4])
                functions.evaluate(pos_map, x, sess, 100, 110, useGTData=True, plot=False)
            
evaluate(False)
