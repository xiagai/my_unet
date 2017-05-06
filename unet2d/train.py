'''
Created on Apr 19, 2017

@author: xiagai
'''
import tensorflow as tf
from unet2d import functions


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '//houseware/codalab/Liver_Tumor_Segmentation_Challenge/experiment/training_result6',
                           """Directory where to write event logs"""
                           """and checkpoint.""")


getTrainDataMachine = functions.GetRandomData(lower=0, upper=100)

def train():
    global_step = tf.contrib.framework.get_or_create_global_step()
    x = tf.placeholder(tf.float32, [512, 512])
    y = tf.placeholder(tf.float32, [512, 512, 2])
    image = tf.reshape(x, [1, 512, 512, 1])
    label = tf.reshape(y, [1, 512, 512, 2])
    
    logits = functions.inference(image, 32)
    pos_map = functions.softmax(logits)
    loss = functions.loss(pos_map, label)
    train_op = functions.train(loss, global_step)
    
    saver = tf.train.Saver(max_to_keep=20)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(40000):
            img, lab = functions.inputs(useGTData=True, weighted_label=1, getTrainDataMachine=getTrainDataMachine)
            sess.run(train_op, feed_dict={x: img, y: lab})
            if i % 1000 == 0 and i != 0:
                functions.evaluate(pos_map, x, sess, 100, 110, useGTData=True, plot=False, threshold=0.2)
            if i % 2000 == 0 and i != 0:
                saver.save(sess, FLAGS.train_dir + '/model%d.ckpt' %i)
            if i % 10 == 0:
                print("step %d, loss: %.4f" %(i, loss.eval(feed_dict={x: img, y: lab}, session=sess)))
            # print(global_step.eval(session=sess))
        saver.save(sess, FLAGS.train_dir + '/model.ckpt')
        

def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()
                    
if __name__ == '__main__':
    tf.app.run()
                    