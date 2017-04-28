'''
Created on Apr 20, 2017

@author: xiagai
'''
import tensorflow as tf
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
        y = tf.placeholder(tf.int64, [512, 512])
        
        image = tf.reshape(x, [1, 512, 512, 1])
        label = tf.reshape(y, [512, 512])
        
        logits = functions.inference(image, 32)
        pos_map = tf.nn.softmax(logits)
        prediction_map = tf.reshape(tf.argmin(pos_map, axis=3), [512, 512])
        correct_prediction = tf.equal(prediction_map, label)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        safer = tf.train.Saver()
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                safer.restore(sess, ckpt.model_checkpoint_path)
            
            total_accuracy = 0
            total_jc = 0
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
                acc = accuracy.eval(feed_dict={x: img_test, y: lab_test}, session=sess)
                
                jaccard = metric.jc(lab_test, pos_array_threshold)
                '''
                total_jc += jc
                total_accuracy += acc
                
                if i % 100 == 0 and i != 0:
                    print("accuracy: %.4f, jc: %.4f" % (total_accuracy / (i + 1), (total_jc / (i + 1))))
                '''
                print("acc: %.4f" %acc)
                print("jaccard: %.4f" % jaccard)
evaluate()
