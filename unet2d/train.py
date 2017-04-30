'''
Created on Apr 19, 2017

@author: xiagai
'''
import tensorflow as tf
from unet2d import functions
from unet2d import metric
import numpy as np


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '//houseware/codalab/Liver_Tumor_Segmentation_Challenge/experiment/training_result',
                           """Directory where to write event logs"""
                           """and checkpoint.""")

def validate(pos_map, x, sess):
    scores = {}
    scores['dc'] = 0.
    scores['jc'] = 0.
    scores['ravd'] = 0.
    scores['assd'] = 0.
    scores['hd'] = 0.
    
    for j in range(500):
        img_test, lab_test = functions.inputs(isTestData=True, useGTData=True, weighted_label=0, randomly=False)
        pos_array = pos_map.eval(feed_dict={x: img_test}, session=sess)[0, :, :, 0]
        pos_array_threshold = pos_array.copy()
        pos_array_threshold[pos_array_threshold > 0.5] = 1
        pos_array_threshold[pos_array_threshold <= 0.5] = 0
        
        dc = metric.dice(pos_array_threshold, lab_test)
        jc = metric.jaccard(pos_array_threshold, lab_test)
        ravd = metric.ravd(pos_array_threshold, lab_test)
        if 0 == np.count_nonzero(pos_array_threshold) or 0 == np.count_nonzero(lab_test):
            assd = scores['assd'] / (j + 1)
            hd = scores['hd'] / (j + 1)
        else:
            assd = metric.assd(pos_array_threshold, lab_test)
            hd = metric.hausdorff(pos_array_threshold, lab_test)
        scores['dc'] += dc
        scores['jc'] += jc
        scores['ravd'] += ravd
        scores['assd'] += assd
        scores['hd'] += hd
        
        if j % 100 == 0 and j != 0:
            for k, v in scores.items():
                print(k + ': ', end='')
                print((v / (j + 1)), end='    ')
            print('\n')


def train():
    x = tf.placeholder(tf.float32, [512, 512])
    y = tf.placeholder(tf.float32, [512, 512, 2])
    image = tf.reshape(x, [1, 512, 512, 1])
    label = tf.reshape(y, [1, 512, 512, 2])
    
    logits = functions.inference(image, 32)
    pos_map = functions.softmax(logits)
    loss = functions.loss(pos_map, label)
    train_op = functions.train(loss)
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            img, lab = functions.inputs(isTestData=False, useGTData=True, weighted_label=2, randomly=True)
            sess.run(train_op, feed_dict={x: img, y: lab})
            if i % 1000 == 0 and i != 0:
                validate(pos_map, x, sess)
            if i % 10 == 0:
                print("step %d, loss: %.4f" %(i, loss.eval(feed_dict={x: img, y: lab}, session=sess)))
        saver.save(sess, FLAGS.train_dir)
        

def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()
                    
if __name__ == '__main__':
    tf.app.run()
                    