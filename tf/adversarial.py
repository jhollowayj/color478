# This is essentially attempt 2 of trying to get a NN to colorize photos.
# Instead of using basic Euclidean loss, I'm attempting to use GAN
# http://arxiv.org/abs/1406.2661
# Plus some other tricks.
import tensorflow as tf
import numpy as np
import ops
import data
import common
from classifier import Classifier


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_data', "/home/ryan/data/lil/lil-train.txt", 'where to find training images')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 6, 'Batch size')

lambda_feat = 1e0
lambda_adv = 1e-1
lambda_img = 1e1
    
class Generator(common.Model):
    def __init__(self, phase_train, input_rgb, truth_rgb):
        self.input_rgb = input_rgb
        self.truth_rgb = truth_rgb
        self.classifier = Classifier()
        common.Model.__init__(self, phase_train=phase_train)

    def build_comparator(self):
        feat = self.classifier.feat
        feat_gen, feat_data = tf.split(0, 2, feat)
        self.loss_feat = lambda_feat * ops.l2_dist_squared(feat_gen, feat_data)
        tf.scalar_summary("loss_feat", self.loss_feat)

    def build_discriminator(self):
        input_imgs = tf.concat(0, [self.gen_rgb, self.truth_rgb])
        input_labels = tf.concat(0, [tf.zeros([FLAGS.batch_size]), tf.ones([FLAGS.batch_size])])

        #self.discr_features = self.classifier.features("discr_classifier", input_imgs)
        #features_bn = self.bn_list(self.discr_features)
        x = input_imgs
        x = self.conv("p0", x, out_chans=32, strides=2)
        x = self.conv("p1", x, out_chans=64, strides=2)
        x = self.conv("p2", x, out_chans=128, strides=2)
        x = self.conv("p3", x, out_chans=256, strides=2)
        x = self.conv("p4", x, out_chans=512, strides=2)

        print "discr shape before avg pool", x.get_shape()

        x = tf.reduce_mean(x, [1,2], keep_dims=False)
        d = common._depth(x)

        p_weight = tf.get_variable("p_weight", [d,1],
            initializer=tf.truncated_normal_initializer())
        p_bias = tf.get_variable("p_bias", [1],
            initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(tf.matmul(x, p_weight), p_bias)
        assert x.get_shape().as_list() == [ 2 * FLAGS.batch_size, 1]
        x = tf.squeeze(x)

        #x = self.maybe_dropout(x, 0.5)

        prob = tf.nn.sigmoid(x)

        # Calculate loss

        prob_gen, prob_data = tf.split(0, 2, prob)

        self.prob_gen = tf.reduce_mean(prob_gen)
        self.prob_data = tf.reduce_mean(prob_data)
        tf.scalar_summary("prob_gen", self.prob_gen)
        tf.scalar_summary("prob_data", self.prob_data)

        real = ops.binary_cross_entropy_with_logits(tf.ones_like(prob_data), prob_data)
        fake = ops.binary_cross_entropy_with_logits(tf.zeros_like(prob_gen), prob_gen)
        self.loss_discr = real + fake
        tf.scalar_summary("loss_discr", self.loss_discr)

        self.loss_adv = lambda_adv * ops.binary_cross_entropy_with_logits(tf.ones_like(prob_gen), prob_gen)
        tf.scalar_summary("loss_adv", self.loss_adv)


    def build_generator(self):
        x = self.input_rgb

        # First get the feature vectors from classifier
        features = self.classifier.features("gen_classifier", x)

        # Then batch norm all those features
        features_bn = self.bn_list(features)

        # Now we start the residual encoder thing

        def conv(name, x, out_chans, activation=tf.nn.relu):
            return self.conv(name, x, shape=3, strides=1, out_chans=out_chans, activation=activation)

        def gate(name, classifier_layer, color_info):
            return classifier_layer + color_info

        x = features_bn[4]
        x = conv("color4", x, 512)

        w = common._width(features_bn[3])
        x = tf.image.resize_bilinear(x, [ w, w ])
        x = gate("gate3", features_bn[3], x)
        x = conv("color3", x, 256)

        w = common._width(features_bn[2])
        x = tf.image.resize_bilinear(x, [ w, w ])
        x = gate("gate2", features_bn[2], x)
        x = conv("color2", x, 64)
        
        w = common._width(features_bn[1])
        x = tf.image.resize_bilinear(x, [ w, w ])
        x = gate("gate1", features_bn[1], x)
        x = conv("color1", x, 3)

        w = common._width(features_bn[0])
        x = tf.image.resize_bilinear(x, [ w, w ])
        x = gate("gate0", features_bn[0], x)
        x = conv("color0", x, 3, tf.nn.relu)

        x = conv("uv", x, 2, tf.nn.sigmoid)

        assert common._width(self.input_rgb) == common._width(x)
        gen_uv = x

        gray = ops.desaturate(self.input_rgb)
        self.gen_rgb = ops.rgb_inference(gray, gen_uv)

        self.loss_img = lambda_img * ops.blur_uv_loss(self.truth_rgb, self.gen_rgb)
        #self.loss_img = lambda_img * ops.l2_dist_squared(self.truth_rgb, self.gen_rgb)
        tf.scalar_summary("loss_img", self.loss_img)

    def build(self):
        with tf.variable_scope("gen"):
            self.build_generator()

        with tf.variable_scope("discr"):
            self.build_discriminator()

        self.build_comparator()


        self.loss_gen = self.loss_feat +  \
                        self.loss_adv +  \
                        self.loss_img
        tf.scalar_summary("loss_gen", self.loss_gen)

        summary_image = ops.build_summary_image(self.input_rgb, self.gen_rgb, self.truth_rgb, True)
        tf.image_summary("s", summary_image, max_images=6)

    def train_prep(self, sess):
        self.gen_train_op, self.gen_saver, self.gen_latest_filename = \
                self.train_prep_subnetwork("gen", sess, self.loss_gen)
        self.discr_train_op, self.discr_saver, self.discr_latest_filename = \
                self.train_prep_subnetwork("discr", sess, self.loss_discr)

        self.summary_writer = tf.train.SummaryWriter('log')
        self.summary_op = tf.merge_all_summaries()
        sess.run(tf.initialize_all_variables())

    def train_prep_subnetwork(self, name, sess, loss):
        variables = []
        for var in tf.trainable_variables():
            if var.name.startswith(name + "/"):
                variables.append(var)
                       
        opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        grads_and_vars = opt.compute_gradients(loss, variables)
        for grad, var in grads_and_vars:
            tf.histogram_summary(var.op.name, var)
            if grad:
                tf.histogram_summary(var.op.name + '/grads', grad)
        train_op = opt.apply_gradients(grads_and_vars)
        
        saver = tf.train.Saver(variables + [ self.step ])

        # Try to restore checkpoint
        latest_filename = name + '-checkpoint'
        checkpoint = tf.train.latest_checkpoint('.', latest_filename)
        if checkpoint:
            print "restoring from checkpoint", checkpoint
            saver.restore(sess, checkpoint)
        else:
            print "can't find %s ... starting from scratch" % latest_filename

        return train_op, saver, latest_filename

    def train_step(self, sess, feed_dict):
        step = sess.run(self.inc_step)
        save_summary = (step % 10 == 0)

        out = { 'step': step }

        # train discriminator
        i = [ self.discr_train_op, self.loss_discr, self.prob_gen, self.prob_data ]
        o = sess.run(i, feed_dict)
        out['loss_discr'] = o[1]
        out['prob_gen'] = o[2]
        out['prob_data'] = o[3]

        # train geneartor
        i = [ self.gen_train_op, self.loss_gen, self.loss_feat, self.loss_adv, self.loss_img ]
        o = sess.run(i, feed_dict)
        out['loss_gen']  = o[1]
        out['loss_feat'] = o[2]
        out['loss_adv']  = o[3]
        out['loss_img']  = o[4]

        if save_summary:
            summary_str = sess.run(self.summary_op, feed_dict)
            self.summary_writer.add_summary(summary_str, step)

            self.gen_saver.save(sess, self.gen_latest_filename, global_step=step,
                latest_filename=self.gen_latest_filename)

            self.discr_saver.save(sess, self.discr_latest_filename, global_step=step,
                latest_filename=self.discr_latest_filename)

        return out
        


    def bn_list(self, features):
        features_bn = []
        i = 0
        for f in features:
            bn = self.batch_norm("feat_bn_%d" % i, f) 
            features_bn.append(bn)
            i += 1
        return features_bn




def train():
    dataset = data.DataSet(FLAGS.train_data)

    input_rgb = tf.placeholder("float", [FLAGS.batch_size, 224, 224, 3], name="input_rgb")
    truth_rgb = tf.placeholder("float", [FLAGS.batch_size, 224, 224, 3], name="truth_rgb")
    phase_train = tf.placeholder("bool", name="phase_train")

    model = Generator(phase_train=phase_train, input_rgb=input_rgb, truth_rgb=truth_rgb)

    sess = tf.Session()
    model.train_prep(sess)

    while True:
        batch = dataset.next_batch(FLAGS.batch_size)
        desaturated_batch = np_desaturate(batch)

        o = model.train_step(sess, {
            input_rgb: desaturated_batch,
            truth_rgb: batch,
            phase_train: True,
        })

        print "%d: loss G/D %.04f/%.04f prob G/D %.02f/%.02f" % (o['step'], o['loss_gen'], o['loss_discr'], \
                o['prob_gen'], o['prob_data'])
        print "loss_feat = %.04f" % o['loss_feat']
        print "loss_adv = %.04f" % o['loss_adv']
        print "loss_img = %.04f" % o['loss_img']


def np_desaturate(batch):
    r = batch[:, :, :, 0]
    g = batch[:, :, :, 1]
    b = batch[:, :, :, 2]
    gray = (r + g + b ) / 3.0
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    assert gray_rgb.shape == batch.shape
    return gray_rgb


def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()
