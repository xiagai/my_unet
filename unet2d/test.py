'''
Created on Apr 20, 2017

@author: xiagai
'''
import tensorflow as tf
import numpy as np
from unet2d import functions
import matplotlib.pyplot as plt
from unet2d import metric




FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/houseware/codalab/Liver_Tumor_Segmentation_Challenge/training_result2',
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
        prediction_map = tf.reshape(tf.argmin(pos_map, axis=3), [512, 512])
        
        safer = tf.train.Saver()
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                safer.restore(sess, ckpt.model_checkpoint_path)
            
            scores = {}
            scores['dc'] = 0.
            scores['jc'] = 0.
            scores['ravd'] = 0.
            scores['assd'] = 0.
            scores['hd'] = 0.
            
            for i in range(2000):
                img_test, lab_test = functions.inputs(isTestData=True, useGTData=True, weighted_label=False, randomly=False)
                prediction = prediction_map.eval(feed_dict={x: img_test}, session=sess)
                pos_array = pos_map.eval(feed_dict={x: img_test}, session=sess)[0, :, :, 0]
                pos_array_threshold = pos_array.copy()
                pos_array_threshold[pos_array_threshold > 0.2] = 1
                pos_array_threshold[pos_array_threshold <= 0.2] = 0
                # plot part
                '''
                plt.subplot(3, 2, 1)
                plt.imshow(img_test, cmap='gray')
                plt.subplot(3, 2, 2)
                plt.imshow(lab_test, cmap='gray')
                plt.subplot(3, 2, 3)
                plt.imshow(prediction, cmap='gray')
                plt.subplot(3, 2, 4)
                plt.imshow(pos_array, cmap='gray')
                plt.subplot(3, 2, 5)
                plt.imshow(pos_array_threshold, cmap='gray')
                plt.show()
                '''
                
                dc = metric.dice(pos_array_threshold, lab_test)
                jc = metric.jaccard(pos_array_threshold, lab_test)
                ravd = metric.ravd(pos_array_threshold, lab_test)
                if 0 == np.count_nonzero(pos_array_threshold) or 0 == np.count_nonzero(lab_test):
                    assd = scores['assd'] / (i + 1)
                    hd = scores['hd'] / (i + 1)
                else:
                    assd = metric.assd(pos_array_threshold, lab_test)
                    hd = metric.hausdorff(pos_array_threshold, lab_test)
                scores['dc'] += dc
                scores['jc'] += jc
                scores['ravd'] += ravd
                scores['assd'] += assd
                scores['hd'] += hd
                
                if i % 100 == 0 and i != 0:
                    for k, v in scores.items():
                        print(k + ': ', end='')
                        print((v / (i + 1)), end='    ')
                    print('\n')
evaluate()
