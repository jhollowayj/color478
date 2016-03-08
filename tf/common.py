from tensorflow.python.ops import control_flow_ops
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def _depth(x):
    return x.get_shape().as_list()[-1]

def _width(x):
    return x.get_shape().as_list()[2]

class Model():
    def __init__(self, phase_train):
        self.phase_train = phase_train
        self.latest_filename = "gen-checkpoint"

        self.step = tf.Variable(0, trainable=False, name="step")
        self.inc_step = self.step.assign_add(1) 
        self.build()

    def maybe_dropout(self, x, amount=0.5):
        return control_flow_ops.cond(self.phase_train,
                lambda: tf.nn.dropout(x, amount),
                lambda: x)

    def batch_norm(self, name, x):
        d = _depth(x)
        #with tf.variable_op_scope([x], name, "bn"):
        with tf.variable_scope(name):
            beta = tf.get_variable('beta', [d], initializer=tf.constant_initializer(0.0))
            gamma = tf.get_variable('gamma', [d], initializer=tf.constant_initializer(1.0))

            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            ema = tf.train.ExponentialMovingAverage(decay=0.9, num_updates=self.step)
            ema_apply_op = ema.apply([batch_mean, batch_var])
            ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

            def mean_var_with_update():
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = control_flow_ops.cond(self.phase_train,
                mean_var_with_update,
                lambda: (ema_mean, ema_var))

            normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
                beta, gamma, 1e-3, scale_after_normalization=True)

            return normed

    def conv(self, name, x, out_chans, shape=3, strides=1, activation=tf.nn.relu):
        in_chans = _depth(x)
        with tf.variable_scope(name):
            kernel = tf.get_variable('kernel', [shape, shape, in_chans, out_chans],
                initializer=tf.truncated_normal_initializer())
            x = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME')
            x = self.batch_norm('bn', x)
            x = activation(x)
            return x

    def resnet_unit(self, name, x, out_chans, shape=3, strides=1, last_activation=tf.nn.relu):
        skip = x
        in_chans = x.get_shape().as_list()[3]
        _strides = [1, strides, strides, 1]

        with tf.variable_scope(name):
            kernelA_shape = [shape, shape, in_chans, out_chans]
            kernelA = tf.get_variable("kernelA", kernelA_shape,
                 initializer=tf.truncated_normal_initializer())
            x = tf.nn.conv2d(x, kernelA, _strides, padding="SAME")
            x = self.batch_norm("bnA", x)
            x = tf.nn.relu(x)

            kernelB_shape = [shape, shape, out_chans, out_chans]
            kernelB = tf.get_variable("kernelB", kernelB_shape,
                 initializer=tf.truncated_normal_initializer())
            x = tf.nn.conv2d(x, kernelB, [1,1,1,1], padding="SAME")
            x = self.batch_norm("bnB", x)

            # Path 2: Identity / skip connection
            if strides > 1:
                skip = tf.nn.avg_pool(skip, [1,shape,shape,1], [1,strides,strides,1], padding='SAME')

            if out_chans > in_chans:
                skip = tf.pad(skip, [[0,0], [0,0], [0,0], [0, out_chans - in_chans]])
            elif out_chans < in_chans:
                # learn a projection 1x1 conv
                projection_shape = [1, 1, in_chans, out_chans]
                projection = tf.get_variable("projection", projection_shape, initializer=tf.truncated_normal_initializer())
                skip = tf.nn.conv2d(skip, projection, [1,1,1,1], padding="SAME")

            x = x + skip
            x = last_activation(x)
            return x


    def build(self):
        # Must set self.loss
        raise NotImplementedError

    def print_flags(self):
        flags = FLAGS.__dict__['__flags']
        for f in flags:
            print f, flags[f]

