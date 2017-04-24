'''
Created on Apr 19, 2017

@author: xiagai
'''
import tensorflow as tf
from unet2d import functions
import time


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '//houseware/codalab/Liver_Tumor_Segmentation_Challenge/training_result',
                           """Directory where to write event logs"""
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        x = tf.placeholder(tf.float32, [512, 512])
        y = tf.placeholder(tf.float32, [512, 512, 2])
        image = tf.reshape(x, [1, 512, 512, 1])
        label = tf.reshape(y, [1, 512, 512, 2])
        
        logits = functions.inference(image)
        loss = functions.loss(logits, label)
        train_op = functions.train(loss, global_step)
        
        
        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""
            
            def begin(self):
                self._step = -1
                
            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss) # Asks for loss value
            
            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 10 == 0:
                    sec_per_instance = float(duration)
                
                    format_str = ('step %d, loss = %.2f (%.3f '
                                  'sec/instance)')
                    print(format_str % (self._step, loss_value, sec_per_instance))
                    
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                    tf.train.NanTensorHook(loss),
                    _LoggerHook()],
            config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                img, lab = functions.inputs(isTestData=False, useGTData=True, randomly=True)
                mon_sess.run(train_op, feed_dict={x: img, y: lab})
                    
        
def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()
                    
if __name__ == '__main__':
    tf.app.run()
                    