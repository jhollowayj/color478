#!/usr/bin/env python

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import tensor_shape

import numpy as np
from sys import exit
import csv
import signal
import datetime
import os
import time
import data
import ops
import vgg16
from skimage.io import imsave

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_data', None, 'where to find training images')
flags.DEFINE_string('val_data', None, 'validation data set')

flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 2, 'Batch size')
flags.DEFINE_boolean('train_color', False, 'train on color images in addition to bw images')
flags.DEFINE_boolean('list_vars', False, 'list variables used')
flags.DEFINE_boolean('bn_debug', False, 'print bn variables for debugging')

flags.DEFINE_string('gen_checkpoint', None, 'load generator from stored checkpoint')
flags.DEFINE_string('disc_checkpoint', None, 'load discriminator from stored checkpoint')

flags.DEFINE_string('train_mode', 'both', 'which models to train. allowed values "gen", "disc", "both" (default)')
flags.DEFINE_string('forward', None, 'forward mode. filename to input image')
flags.DEFINE_boolean('save', False, 'must use with --forward. saves a complete colorize.tfmodel file')

with open("../tensorflow-vgg16/vgg16-20160129.tfmodel", mode='rb') as f:
    fileContent = f.read()
vgg16_graph_def = tf.GraphDef()
vgg16_graph_def.ParseFromString(fileContent)

def is_train_gen():
    return FLAGS.train_mode == 'both' or FLAGS.train_mode == 'gen'

def is_train_disc():
    return FLAGS.train_mode == 'both' or FLAGS.train_mode == 'disc'

def depth(layer):
    return layer.get_shape().as_list()[3]

def width(layer):
    return layer.get_shape().as_list()[2]

class CommonModel():
    def __init__(self, name, phase_train, step, copy_from=None, restore_sess=None):
        self.name = name
        self.latest_filename = self.name + "-checkpoint"
        self.phase_train = phase_train
        self.step = step
        self.restore_vars = { "step": step }
        self.trainable_vars = []
        self.trainable_var_names = []
        self.saver = None
        self.restore_sess = restore_sess
        self.copy_from = copy_from

    def var_name(self, *args):
        l = list(args)
        if not args[0].startswith(self.name):
            l.insert(0, self.name)
        return '/'.join(l)

    def get_variable(self, name, shape, initializer, trainable=True):
        shape = tensor_shape.as_shape(shape)
        dtype = tf.float32

        if self.copy_from:
            var_to_restore = self.copy_from.restore_vars[name]
            value = var_to_restore.eval(self.restore_sess)
            return tf.constant(value, name=name) #maybe need dtype here

        if name in self.restore_vars:
            v = self.restore_vars[name]
            if trainable:
                assert name in self.trainable_var_names
                assert v in self.trainable_vars   
            return v

        # otherwise create variable

        # Clear control dependencies while creating the initializer ops.
        with tf.control_dependencies(None):
            with tf.name_scope(name + "/Initializer/"):
                init_val = initializer(shape, dtype=dtype)
            v = tf.Variable(init_val,  name=name, trainable=trainable)

        self.restore_vars[name] = v

        if trainable:
            self.trainable_vars.append(v)
            self.trainable_var_names.append(name)

        return v


    def train_op(self, loss):
        assert loss

        if FLAGS.list_vars:
            print "LISTING VARS FOR", self.name
            for name in self.restore_vars:
                if name in self.trainable_var_names:
                    print "T", name 
                else:
                    print name 

        opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.0)
        grads_and_vars = opt.compute_gradients(loss, self.trainable_vars)

        if self.name == "gen":
            for grad, var in grads_and_vars:
                if var.name == "gen/uv/filter:0":
                    self.uv_filter_grad = grad
                    #tf.histogram_summary("uv_filter_grad", uv_filter_grad)
                else:
                    print var.name

        elif self.name == "disc":
            for grad, var in grads_and_vars:
                # Add histograms for trainable variables.
                tf.histogram_summary(var.op.name, var)
                # Add histograms for gradients.
                if grad:
                    pass
                    #tf.histogram_summary(var.op.name + '/gradients', grad)

        op = opt.apply_gradients(grads_and_vars)

        if not self.saver:
            self.saver = tf.train.Saver(self.restore_vars)

        return op

    def maybe_restore_checkpoint(self, sess, checkpoint):
        if not self.saver:
            print "no saver for ", self.name
            return
        if checkpoint:
            print "flag used. restoring from checkpoint", checkpoint, self.latest_filename
            self.saver.restore(sess, checkpoint)
        else:
            checkpoint = tf.train.latest_checkpoint('.', self.latest_filename)
            if checkpoint:
                print "restoring from checkpoint", checkpoint
                self.saver.restore(sess, checkpoint)
            else:
                print "cant find %s ... starting generator from scratch" % (self.latest_filename)
        return checkpoint

    def save(self, sess, step):
        self.saver.save(sess, self.latest_filename, global_step=step,
            latest_filename=self.latest_filename)

    def activated_summary(self, name, layer):
        nonzero = tf.select(layer > 0, tf.ones_like(layer), tf.zeros_like(layer))
        count = tf.reduce_sum(nonzero)
        size = tf.to_float(tf.size(layer))
        tf.scalar_summary(name, count/size)

    def maybe_dropout(self, x, amount):
        return
        #return control_flow_ops.cond(self.phase_train,
        #        lambda: tf.nn.dropout(x, amount),
        #        lambda: x)

    def resnet_unit(self, name, x, out_chans, shape=3, strides=1):
        skip = x
        in_chans = x.get_shape().as_list()[3]

        _strides = [1, strides, strides, 1]

        filterA_name = self.var_name(name, "filterA")
        filterA_shape = [shape, shape, in_chans, out_chans]
        filterA = self.get_variable(filterA_name, filterA_shape, tf.truncated_normal_initializer())

        x = tf.nn.conv2d(x, filterA, _strides, padding="SAME")

        x = self.batch_norm(name + "_bnA", x, scale_after_norm=False)

        x = tf.nn.relu(x)

        filterB_name = self.var_name(name, "filterB")
        filterB_shape = [shape, shape, out_chans, out_chans]
        filterB = self.get_variable(filterB_name, filterB_shape, tf.truncated_normal_initializer())

        x = tf.nn.conv2d(x, filterB, [1,1,1,1], padding="SAME")

        x = self.batch_norm(name + "_bnB", x, scale_after_norm=True)

        # Path 2: Identity / skip connection
        if strides > 1:
            skip = tf.nn.avg_pool(skip, [1,shape,shape,1], [1,strides,strides,1], padding='SAME')

        if out_chans > in_chans:
            skip = tf.pad(skip, [[0,0], [0,0], [0,0], [0, out_chans - in_chans]])

        if out_chans < in_chans:
            # learn a projection 1x1 conv
            projection_shape = [1, 1, in_chans, out_chans]
            projection = self.get_variable(self.var_name(name, "projection"), projection_shape, tf.truncated_normal_initializer())
            skip = tf.nn.conv2d(skip, projection, [1,1,1,1], padding="SAME")


        x = x + skip

        x = tf.nn.relu(x)

        return x

    def conv_layer(self, name, a, shape, strides, out_chans, activation=tf.nn.relu, use_bias=False):
        in_chans = a.get_shape().as_list()[3]

        filter_name = self.var_name(name, "filter")

        filter_shape = [shape, shape, in_chans, out_chans]

        filt = self.get_variable(filter_name, filter_shape, tf.truncated_normal_initializer(stddev=0.1))
        if name == "color0":
            tf.histogram_summary(filter_name, filt)

        a = tf.nn.conv2d(a, filt, [1,strides,strides,1], padding="SAME")
        

        if use_bias:
            bias = self.get_variable(self.var_name(name, "bias"), [out_chans], tf.constant_initializer(0.0))
            a = tf.nn.bias_add(a, bias)
        else:
            scale_after_norm = not (activation == tf.nn.relu)
            a = self.batch_norm(name + "_bn", a, scale_after_norm)
  
        out = activation(a)

        return out

    def batch_norm(self, bn_name, x, scale_after_norm=True):
        shape = x.get_shape().as_list()

        assert len(shape) == 4
        depth = shape[-1]
        norm_shape = [depth]

        beta = self.get_variable(self.var_name(bn_name, "beta"), norm_shape,
                tf.constant_initializer(0.0))
        gamma = self.get_variable(self.var_name(bn_name, "gamma"), norm_shape,
                tf.constant_initializer(1.0), trainable=scale_after_norm)

        batch_mean, batch_variance = tf.nn.moments(x, [0, 1, 2])
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_variance])

        mean_v = self.get_variable(self.var_name(bn_name, "mean"), norm_shape,
                tf.constant_initializer(0.0), trainable=False)
        variance_v = self.get_variable(self.var_name(bn_name, "variance"), norm_shape,
                tf.constant_initializer(1.0), trainable=False)


        def mean_var_train():
            if mean_v.__class__ == tf.Tensor:
                # This happens in the case where we're building a const graph.
                # It crashes when the code below tries to assign to a tensor
                # TF traverses each path when creating the graph.
                return mean_v, variance_v
            assert mean_v.__class__ == tf.Variable

            with tf.control_dependencies([ ema_apply_op ]):
                with tf.control_dependencies([ 
                    mean_v.assign(ema.average(batch_mean)),
                    variance_v.assign(ema.average(batch_variance))
                ]):
                    return tf.identity(batch_mean), tf.identity(batch_variance)

        def mean_var_test():
            return mean_v, variance_v

        mean, var = control_flow_ops.cond(self.phase_train, mean_var_train, mean_var_test)

        eps = 1e-12

        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
            beta, gamma, eps, scale_after_norm)

        return normed


    def maybe_load_vgg16(self):
        g = tf.get_default_graph()
        loaded = False
        for op in g.get_operations():
            if op.name.startswith("vgg16/"):
                loaded = True
                break

        if not loaded:
            tf.import_graph_def(vgg16_graph_def, name="vgg16")

    def vgg_top(self, rgb, name):
        g = tf.get_default_graph()

        self.maybe_load_vgg16()
                
        import_name = "vgg16_" + name

        vgg = VGGCopyFromImport(import_name)
        vgg.build(rgb)
        assert vgg.prob

        pool0_bn = self.batch_norm(self.var_name("pool0/bn"), rgb)

        pool1 = g.get_operation_by_name(import_name + "/pool1").inputs[0]
        pool1_bn = self.batch_norm(self.var_name("pool1/bn"), pool1)

        pool2 = g.get_operation_by_name(import_name + "/pool2").inputs[0]
        pool2_bn = self.batch_norm(self.var_name("pool2/bn"), pool2)

        pool3 = g.get_operation_by_name(import_name + "/pool3").inputs[0]
        pool3_bn = self.batch_norm(self.var_name("pool3/bn"), pool3)

        pool4 = g.get_operation_by_name(import_name + "/pool4").inputs[0]
        pool4_bn = self.batch_norm(self.var_name("pool4/bn"), pool4)

        #top = [ rgb, pool1, pool2, pool3, pool4 ] 
        top = [ pool0_bn, pool1_bn, pool2_bn, pool3_bn, pool4_bn ] 

        return top

class VGGCopyFromImport(vgg16.Model):
    def __init__(self, name):
        self.name = name

    def build(self, rgb):
        g = tf.get_default_graph()
        with g.name_scope(self.name):
            return vgg16.Model.build(self, rgb)

    def get_conv_filter(self, name):
        g = tf.get_default_graph()
        t = g.get_tensor_by_name("vgg16/" + name + "/filter:0")
        assert t
        return t

    def get_bias(self, name):
        g = tf.get_default_graph()
        t = g.get_tensor_by_name("vgg16/" + name + "/bias:0")
        assert t
        return t

    def get_fc_weight(self, name):
        g = tf.get_default_graph()
        t = g.get_tensor_by_name("vgg16/" + name + "/weight:0")
        assert t
        return t


class Generator(CommonModel):
    def __init__(self, truth_rgb, input_rgb, phase_train, step=None, copy_from=None, restore_sess=None):
        CommonModel.__init__(self, "gen", phase_train=phase_train, step=step,
                copy_from=copy_from, restore_sess=restore_sess)

        self.truth_rgb = truth_rgb
        self.input_rgb = input_rgb
        self.gray = ops.desaturate(self.truth_rgb)
        self.gray3 = tf.concat(3, [self.gray, self.gray, self.gray],
                name=self.var_name("gray3"))

        self.build_model()


    def build_model(self):
        inferred_color = self.residue_encoder(self.input_rgb)
        #inferred_color = self.other_encoder(self.input_rgb)
        self.inferred_rgb = ops.rgb_inference(self.gray, inferred_color,
                name=self.var_name("inferred_rgb"))
        summary_image = ops.build_summary_image(self.gray3, self.inferred_rgb, self.truth_rgb, False)
        tf.image_summary("gray", summary_image, max_images=6)

    def other_encoder(self, c):
        c = self.resnet_unit("c_res1",   c, out_chans=64, shape=3, strides=1)
        c = self.resnet_unit("c_res2",   c, out_chans=2, shape=3, strides=1)


        filterA_name = self.var_name("x", "filterA")
        filterA_shape = [3, 3, 64, depth(c), 64]
        filterA = self.get_variable(filterA_name, filterA_shape, tf.truncated_normal_initializer())
        out_shape = [FLAGS.batch_size, 224, 224, 64]
        x = tf.nn.conv2d_transpose(c, filterA, out_shape, [1,2,2,1], padding="SAME")
        print "x shape", x.get_shape().as_list()

        return c


    def residue_encoder(self, rgb):
        d = 1
        top = self.vgg_top(rgb, "gen")

        def activated_summary(n, c):
            return
            self.activated_summary("gen/color" + str(n) + "_activated", c)


        def conv(name, x, out_chans, activation=tf.nn.relu):
            x = self.conv_layer(name, x, 3, 1, out_chans, activation)
            return x

        def gate(name, vgg_layer, color_info):
            return vgg_layer + color_info

        # top 0   224 x 224 x 3
        # top 1   224 x 224 x 64
        # top 2   112 x 112 x 128
        # top 3    56 x  56 x 256
        # top 4    28 x  28 x 512

        c = top[4]
        c = conv("color4", c, 256 / d)

        w = width(top[3])
        c = tf.image.resize_bilinear(c, [ w, w ])
        c = gate("gate3", top[3], c)
        c = conv("color3", c, 128 / d)
        activated_summary(3, c)
        #tf.histogram_summary("color3", c)

        w = width(top[2])
        c = tf.image.resize_bilinear(c, [ w, w ])
        c = gate("gate2", top[2], c)
        c = conv("color2", c, 64 / d)
        activated_summary(2, c)
        #tf.histogram_summary("color2", c)
        
        w = width(top[1])
        c = tf.image.resize_bilinear(c, [ w, w ])
        c = gate("gate1", top[1], c)
        c = conv("color1", c, 3)
        activated_summary(1, c)
        #tf.histogram_summary("color1", c)

        w = width(top[0])
        c = tf.image.resize_bilinear(c, [ w, w ])
        c = gate("gate0", top[0], c)
        c = conv("color0", c, 16, tf.nn.relu)

        c = conv("color0a", c, 8, tf.nn.relu)
        c = conv("uv", c, 2, tf.nn.sigmoid)
        return c


    def construct_const_graph(self, restore_sess):
        const_graph = tf.Graph()
        with const_graph.as_default():
            truth_rgb = tf.placeholder("float", [None, 224, 224, 3], name="truth_rgb")
            input_rgb = tf.placeholder("float", [None, 224, 224, 3], name="input_rgb")
            const_gen = Generator(truth_rgb=truth_rgb, input_rgb=input_rgb,
                phase_train=tf.constant(False), copy_from=self, restore_sess=restore_sess)
        return const_graph
        

def write_graph(graph, save_path="colorize.tfmodel"):
    graph_def = graph.as_graph_def()
    print "graph_def byte size", graph_def.ByteSize()
    graph_def_s = graph_def.SerializeToString()

    with open(save_path, "wb") as f:
      f.write(graph_def_s)
    print "saved model to %s" % save_path



def tf_duplicate_graph(g, original_prefix, new_prefix):
    ops = g.get_operations()
    for op in ops:
        if not op.name.startswith(original_prefix + "/"):
            continue
        print op.name, op.type
        #g.create_op(op.type, inputs, dtypes, input_types=None, name=None, attrs=None, op_def=None, compute_shapes=True)

class Discriminator(CommonModel):
    def __init__(self, step, phase_train):
        CommonModel.__init__(self, "disc", phase_train=phase_train, step=step)


    def get_prob(self, name, c):
        # top 0   224 x 224 x 3
        # top 1   224 x 224 x 64
        # top 2   112 x 112 x 128
        # top 3    56 x  56 x 256
        # top 4    28 x  28 x 512

        c = self.conv_layer("p_conv0", c, shape=7, strides=2, out_chans=64)
        c = tf.nn.max_pool(c, [1,3,3,1], [1,2,2,1], padding='SAME')

        c = self.resnet_unit("p_res1",   c, out_chans=64, shape=3, strides=1)
        #c = self.resnet_unit("p_res1_1", c, out_chans=64, shape=3, strides=1)
        c = self.resnet_unit("p_res2",   c, out_chans=128, shape=3, strides=2)
        #c = self.resnet_unit("p_res2_1", c, out_chans=128, shape=3, strides=1)
        c = self.resnet_unit("p_res3",   c, out_chans=256, shape=3, strides=2)
        #c = self.resnet_unit("p_res3_1", c, out_chans=256, shape=3, strides=1)

        print "get_prob conv output", c.get_shape().as_list()

        w = width(c)
        c = tf.nn.avg_pool(c, [1,w,w,1], [1,1,1,1], padding='VALID')

        c = self.conv_layer("fc_conv", c, shape=1, strides=1, out_chans=1, activation=tf.nn.sigmoid)
        assert c.get_shape().as_list() == [2 * FLAGS.batch_size, 1, 1, 1]
        return tf.squeeze(c)


class Colorizer():
    def __init__(self, sess):
        self.sess = sess

        self.data_set = data.DataSet(FLAGS.train_data)
        if self.data_set.length() < FLAGS.batch_size:
            FLAGS.batch_size = self.data_set.length()

        self.val_data_set = None
        if FLAGS.val_data:
            self.val_data_set = data.DataSet(FLAGS.val_data)

    def bn_debug(self):
        if not FLAGS.bn_debug: return
        self.fc_conv_mean = self.disc.restore_vars["disc/fc_conv_bn/mean"]
        tf.histogram_summary("fc_conv_mean", self.fc_conv_mean)

        self.fc_conv_variance = self.disc.restore_vars["disc/fc_conv_bn/variance"]
        tf.histogram_summary("fc_conv_variance", self.fc_conv_variance)

        color0_mean = self.gen.restore_vars["gen/color0_bn/mean"]
        tf.histogram_summary("color0_mean", color0_mean)

        color0_variance = self.gen.restore_vars["gen/color0_bn/variance"]
        tf.histogram_summary("color0_variance", color0_variance)

    def train(self):
        truth_rgb = tf.placeholder("float", [FLAGS.batch_size, 224, 224, 3], name="truth_rgb")
        input_rgb = tf.placeholder("float", [FLAGS.batch_size, 224, 224, 3], name="input_rgb")
        gen_phase_train = tf.placeholder(tf.bool, name="gen_phase_train")
        disc_phase_train = tf.placeholder(tf.bool, name="disc_phase_train")

        step_var = tf.Variable(0, trainable=False)
        inc_step = step_var.assign_add(1) 

        self.gen = Generator(truth_rgb=truth_rgb, input_rgb=input_rgb,
                phase_train=gen_phase_train, step=step_var)
        self.disc = Discriminator(step=step_var, phase_train=disc_phase_train)

        euclid_loss = self.blur_uv_loss(truth_rgb, self.gen.inferred_rgb)
        tf.scalar_summary("euclid_loss", euclid_loss)

        disc_input = tf.concat(0, [ truth_rgb, self.gen.inferred_rgb ])
        disc_labels = tf.concat(0, [ tf.ones([FLAGS.batch_size]), tf.zeros([FLAGS.batch_size]) ])

        prob = self.disc.get_prob("both", disc_input)
        prob_data, prob_gen = tf.split(0, 2, prob)

        avg_prob_data = tf.reduce_mean(prob_data)
        avg_prob_gen = tf.reduce_mean(prob_gen)

        tf.scalar_summary("prob_data", avg_prob_data)
        tf.scalar_summary("prob_gen", avg_prob_gen)

        disc_loss = ops.binary_cross_entropy_with_logits(disc_labels, prob)
        gen_loss = ops.binary_cross_entropy_with_logits(tf.ones_like(prob_gen), prob_gen)



        tf.scalar_summary("disc_loss", disc_loss)
        tf.scalar_summary("gen_loss", gen_loss)
        tf.scalar_summary("learning_rate", FLAGS.learning_rate)

        train_disc = self.disc.train_op(disc_loss)
        #train_gen_euclid = self.gen.train_op(euclid_loss)
        train_gen = self.gen.train_op(gen_loss)

        self.bn_debug()

        self.sess.run(tf.initialize_all_variables())

        self.gen.maybe_restore_checkpoint(self.sess, FLAGS.gen_checkpoint)
        self.disc.maybe_restore_checkpoint(self.sess, FLAGS.disc_checkpoint)

        summary_writer = tf.train.SummaryWriter('log')
        summary_op = tf.merge_all_summaries()

        if not os.path.isdir('val-imgs'):
            os.mkdir('val-imgs')

        gen_train_csv = TextData('gen-train.csv')
        #val_csv = TextData('val-loss.csv')

        print "begin training"
        print_flags()

        euclid_loss_val = prob_data_val = prob_gen_val = disc_loss_val = gen_loss_val = 0.0

        while True:
            step = self.sess.run(inc_step)

            batch = self.data_set.next_batch(FLAGS.batch_size)
            desaturated_batch = np_desaturate(batch)

            start_time = time.time()

            old_prob_data = prob_data_val
            old_prob_gen = prob_gen_val

            if is_train_disc():
                if old_prob_gen < 0.01 and old_prob_data > 0.5:
                    print "skip training DISC because prob_gen_val < 0.01"
                else:
                    if FLAGS.bn_debug:
                        i = [ train_disc, self.fc_conv_mean, self.fc_conv_variance ]
                    else:
                        i = train_disc

                    o = self.sess.run(i, {
                        truth_rgb: batch,
                        input_rgb: desaturated_batch,
                        gen_phase_train: False,
                        disc_phase_train: True,
                    })

                    if FLAGS.bn_debug:
                        print "fc_conv_mean", o[1], "fc_conv_variance", o[2]

            if is_train_gen():
                if is_train_disc() and old_prob_data < old_prob_gen:
                    print "skip training GEN because prob_data_val < prob_gen_val"
                else:
                    self.sess.run(train_gen, {
                        truth_rgb: batch,
                        input_rgb: desaturated_batch,
                        gen_phase_train: True,
                        disc_phase_train: False,
                    })

#                if FLAGS.train_color:
#                    # Here we make sure that the gen network will output the same color image
#                    # if given a color image as input.
#                    self.sess.run(train_gen_euclid, {
#                        truth_rgb: batch,
#                        input_rgb: batch, # NOTE not desaturated
#                        gen_phase_train: True,
#                    })



            euclid_loss_val, prob_data_val, prob_gen_val, disc_loss_val, gen_loss_val = self.sess.run([
                euclid_loss, avg_prob_data, avg_prob_gen, disc_loss,     gen_loss  ], {
                truth_rgb: batch,
                input_rgb: desaturated_batch,
                gen_phase_train: False,
                disc_phase_train: False,
            })

            if is_train_gen():
                gen_train_csv.write(step, euclid_loss_val, gen_loss_val, disc_loss_val)

            duration = time.time() - start_time

            print('%02d: euclid=%.3f prob data/gen=%.2f/%.2f loss disc/gen=%.2f/%.2f (%.1f sec)' %
                (step, euclid_loss_val, prob_data_val, prob_gen_val, disc_loss_val, gen_loss_val, duration))

            if disc_loss_val > 60:
                print "diverging. exiting"
                exit(1)

            #print "mean (np) ", np.mean(batch, axis=(0,1,2))
            #print "mean (desaturated)", np.mean(np_desaturate(batch), axis=(0,1,2))

            if step % 50 == 0 and self.val_data_set:
                start_time = time.time()
                self.val(step)
                duration = time.time() - start_time

                print('val: (%.1f sec)' % (duration))

            if step % 10 == 0:
                # Update the events file.
                summary_str = self.sess.run(summary_op, feed_dict={
                    truth_rgb: batch,
                    input_rgb: desaturated_batch,
                    gen_phase_train: False,
                    disc_phase_train: False,
                })
                summary_writer.add_summary(summary_str, step)

                if is_train_gen():
                    self.gen.save(self.sess, step)
                if is_train_disc():
                    self.disc.save(self.sess, step)
                     
    def val(self, step):
        const_graph = self.gen.construct_const_graph(self.sess)
        with const_graph.as_default():
            sess = tf.Session() # use a different sess for validation
            sess.run(tf.initialize_all_variables())

            gray3 = const_graph.get_tensor_by_name("gen/gray3:0")
            inferred_rgb = const_graph.get_tensor_by_name("gen/inferred_rgb:0")
            truth_rgb = const_graph.get_tensor_by_name("truth_rgb:0")
            input_rgb = const_graph.get_tensor_by_name("input_rgb:0")
            summary_image = ops.build_summary_image(gray3, inferred_rgb, truth_rgb, True)

            batch = self.val_data_set.next_batch(FLAGS.batch_size)
            desaturated_batch = np_desaturate(batch)
            
            s_color, summary_image_value = sess.run([inferred_rgb, summary_image], {
                truth_rgb: batch,
                input_rgb: desaturated_batch,
            })

            summary_image_index = 0
            for i in range(0, FLAGS.batch_size):
                img = summary_image_value[i]
                summary_image_index += 1
                if summary_image_index <= 4:
                    fn = "val-imgs/val-%06d-%d.jpg" % (step, summary_image_index)
                    imsave(fn, img)

            # This is a sanity check to make sure we're not leaking color
            # info into the inferrence. We provide a desaturated truth
            # and make sure we still get the same answer.
            s_gray = sess.run(inferred_rgb, {
                truth_rgb: desaturated_batch,
                input_rgb: desaturated_batch,
            })
            diff = np.amax(np.abs(s_gray - s_color)) 
            assert diff < 0.0001

    def blur_uv_loss(self, rgb, inferred_rgb):
        uv = ops.rgb2uv(rgb)
        uv_blur0 = ops.rgb2uv(ops.blur(rgb, 3))
        uv_blur1 = ops.rgb2uv(ops.blur(rgb, 5))

        inferred_uv = ops.rgb2uv(inferred_rgb)
        inferred_uv_blur0 = ops.rgb2uv(ops.blur(inferred_rgb, 3))
        inferred_uv_blur1 = ops.rgb2uv(ops.blur(inferred_rgb, 5))

        return ( ops.dist(inferred_uv, uv) +
                 ops.dist(inferred_uv_blur0 , uv_blur0) +
                 ops.dist(inferred_uv_blur1, uv_blur1) ) / 3

def np_desaturate(batch):
    r = batch[:, :, :, 0]
    g = batch[:, :, :, 1]
    b = batch[:, :, :, 2]
    gray = (r + g + b ) / 3.0
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    assert gray_rgb.shape == batch.shape
    return gray_rgb
    

class TextData():
    def __init__(self, fn):
        self.fn = fn
        exists = os.path.isfile(fn)
        self.file = open(fn, 'ab')
        self.writer = csv.DictWriter(self.file, fieldnames=['ts', 'step', 'euclid_loss', 'gen_loss', 'disc_loss'])
        if not exists:
            # print the flags to the csv header.
            flags = FLAGS.__dict__['__flags']
            for f in flags:
                self.file.write("# %s: %s\n" % (f, flags[f]))
            self.file.flush()
            self.writer.writeheader()

    def write(self, step, euclid_loss, gen_loss, disc_loss):
        t = datetime.datetime.now()
        ts = datetime.datetime.isoformat(t)

        self.writer.writerow({
            'ts': ts,
            'step': step,
            'euclid_loss': euclid_loss,
            'gen_loss': gen_loss,
            'disc_loss': disc_loss,
            #'rgb_dist': rgb_dist,
        })
        self.file.flush()





def print_flags():
    flags = FLAGS.__dict__['__flags']
    for f in flags:
        print f, flags[f]


def run_training():
    sess = tf.Session()
    model = Colorizer(sess)
    model.train()
    

def main(_):
    if FLAGS.save:
        forward(FLAGS.forward)
    else:
        run_training()

if __name__ == '__main__':
    tf.app.run()
