'''
Created on Apr 20, 2017

@author: xiagai
'''
import tensorflow as tf
from unet2d import functions
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/houseware/codalab/Liver_Tumor_Segmentation_Challenge/training_result',
                           """Directory where to write event logs"""
                           """and checkpoint.""")

def evaluate():
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [512, 512])
        y = tf.placeholder(tf.int64, [512, 512])
        
        image = tf.reshape(x, [1, 512, 512, 1])
        label = tf.reshape(y, [512, 512])
        
        logits = functions.inference(image)
        pos_map = tf.nn.softmax(logits)
        prediction_map = tf.reshape(tf.argmax(pos_map, axis=3), [512, 512])
        correct_prediction = tf.equal(prediction_map, label)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        safer = tf.train.Saver()
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                safer.restore(sess, ckpt.model_checkpoint_path)
            
            total_accuracy = 0

            for i in range(2000):
                img_test, lab_test = functions.test_inputs()
                plt.subplot(2, 2, 1)
                plt.imshow(img_test, cmap='gray')
                plt.subplot(2, 2, 2)
                plt.imshow(lab_test, cmap='gray')
                prediction = prediction_map.eval(feed_dict={x: img_test}, session=sess)
                plt.subplot(2, 2, 3)
                plt.imshow(prediction, cmap='gray')
                pos_array = pos_map.eval(feed_dict={x: img_test}, session=sess)
                plt.subplot(2, 2, 4)
                plt.imshow(pos_array[0, : , :, 0], cmap='gray')
                plt.show()
                
                acc = accuracy.eval(feed_dict={x: img_test, y: lab_test}, session=sess)
                total_accuracy += acc
                '''
                if i % 100 == 0 and i != 0:
                    print("accuracy: %.4f" % (total_accuracy / i))
                '''
                print(acc)
                    
evaluate()
