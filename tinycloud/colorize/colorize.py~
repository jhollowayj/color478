#!/usr/bin/env python
#import matplotlib
#matplotlib.use('Qt5Agg')
#import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from sys import exit
import csv
import signal
import datetime
import os
import time
import data
from tf_utils import build_summary_image, yuv2rgb, rgb2yuv, blur
from skimage.io import imsave

NUM_AVG = 40

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train', None, 'where to find training images')
flags.DEFINE_string('val', None, 'validation data set')

flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 4, 'Batch size.    '
                                         'Must divide evenly into the dataset sizes.')
flags.DEFINE_boolean('fixed', False, 'run a fixed batch')
flags.DEFINE_boolean('show_input', True, 'show input on summary image')
flags.DEFINE_boolean('load', False, 'load from stored checkpoint')

flags.DEFINE_boolean("hypercolumn_model", False, "use simple hypercolumn model")

flags.DEFINE_string('forward', None, 'forward mode. filename to input image')
flags.DEFINE_boolean('save', False, 'must use with --forward. saves a complete colorize.tfmodel file')

def load_vgg16(images):
    with open("vgg16-v5.tfmodel", mode='rb') as f:
        fileContent = f.read()
    vgg16_graph_def = tf.GraphDef()
    vgg16_graph_def.ParseFromString(fileContent)

    tf.import_graph_def(vgg16_graph_def, input_map={ "images": images }, name="vgg16")

def upsample(value, repeat):
    #return tf.user_ops.upsample(value, repeat=repeat) 
    tensor = tf.convert_to_tensor(value)
    in_shape = tensor.get_shape().as_list()
    h = in_shape[1]
    w = in_shape[2]
    return tf.image.resize_bilinear(tensor, [ h * repeat, w * repeat ])
    
def hypercolumns_inference(rgb, variables_to_restore, train):
    top, top_norm = vgg_top(rgb, variables_to_restore, train)
    
    h0 = top[0] # rgb_norm
    h1 = top[1] # pool 1
    h2 = upsample(top[2], repeat=2) 
    h3 = upsample(top[3], repeat=4) 
    h4 = upsample(top[4], repeat=8) 

    hypercolumns = tf.concat(3, [ h0, h1, h2, h3, h4 ], name="hypercolumns")
    return hypercolumns
    

def vgg_top(rgb, variables_to_restore, restore_sess, train):
    load_vgg16(rgb)

    g = tf.get_default_graph()

    pool1 = g.get_operation_by_name("vgg16/pool1").inputs[0]
    pool1_norm = batch_norm_2d(pool1, "pool1", variables_to_restore, restore_sess, train)

    pool2 = g.get_operation_by_name("vgg16/pool2").inputs[0]
    pool2_norm = batch_norm_2d(pool2, "pool2", variables_to_restore, restore_sess, train)

    pool3 = g.get_operation_by_name("vgg16/pool3").inputs[0]
    pool3_norm = batch_norm_2d(pool3, "pool3", variables_to_restore, restore_sess, train)

    pool4 = g.get_operation_by_name("vgg16/pool4").inputs[0]
    pool4_norm = batch_norm_2d(pool4, "pool4", variables_to_restore, restore_sess, train)

    rgb_norm = rgb - 0.5

    top =    [ rgb, pool1, pool2, pool3, pool4 ] 
    top_norm =    [ rgb_norm, pool1_norm, pool2_norm, pool3_norm, pool4_norm ] 

    return top, top_norm

def depth(layer):
    return layer.get_shape().as_list()[3]

def width(layer):
    return layer.get_shape().as_list()[2]


def rgb_inference(grayscale, color):
    assert depth(grayscale) == 1
    assert depth(color) == 2
    yuv = tf.concat(3, [grayscale, color])
    return yuv2rgb(yuv, name="inferred_rgb")

def color_encoder_hypercolumns(hypercolumns, variables_to_restore, train):
    hypercolumns_shape = hypercolumns.get_shape().as_list()
    print "hypercolumn shape", hypercolumns_shape
    assert hypercolumns_shape[1:3] == [224, 224]

    conv1 = conv_3x3_layer("conv1", hypercolumns, 128, tf.nn.relu, variables_to_restore, train)
    conv2 = conv_3x3_layer("conv2", conv1, 64, tf.nn.relu, variables_to_restore, train)
    conv3 = conv_3x3_layer("conv3", conv2, 3, tf.nn.relu, variables_to_restore, train)
    uv = conv_3x3_layer("uv", conv3, 2, tf.sigmoid, variables_to_restore, train)

    return uv


def color_encoder(top, variables_to_restore, restore_sess, train):
    # top 0 depth 3
    # top 1 depth 64
    # top 2 depth 128
    # top 3 depth 256
    # top 4 depth 512
    total4 = top[4]
    color4 = conv_1x1_layer("color4", total4, 256, tf.nn.relu, variables_to_restore, restore_sess, train)

    width3 = width(top[3])
    resize3 = tf.image.resize_bilinear(color4, [ width3, width3 ])
    total3 = top[3] + resize3
    color3 = conv_3x3_layer("color3", total3, 128, tf.nn.relu, variables_to_restore, restore_sess, train)
    activated_summary("color3_activated", color3)

    width2 = width(top[2])
    resize2 = tf.image.resize_bilinear(color3, [ width2, width2 ])
    total2 = top[2] + resize2
    color2 = conv_3x3_layer("color2", total2, 64, tf.nn.relu, variables_to_restore, restore_sess, train)
    activated_summary("color2_activated", color2)
    
    width1 = width(top[1])
    resize1 = tf.image.resize_bilinear(color2, [ width1, width1 ])
    total1 = top[1] + resize1
    color1 = conv_3x3_layer("color1", total1, 3, tf.nn.relu, variables_to_restore, restore_sess, train)
    activated_summary("color1_activated", color1)

    width0 = width(top[0])
    resize0 = tf.image.resize_bilinear(color1, [ width0, width0 ])
    total0 = top[0] + resize0
    color0 = conv_3x3_layer("color0", total0, 3, tf.nn.relu, variables_to_restore, restore_sess, train)
    activated_summary("color0_activated", color0)

    #drop0 = tf.nn.dropout(color0, 0.5)

    uv = conv_3x3_layer("uv", color0, 2, tf.nn.sigmoid, variables_to_restore, restore_sess, train)

    return uv



def activated_summary(name, layer):
    nonzero = tf.select(layer > 0, tf.ones_like(layer), tf.zeros_like(layer))
    count = tf.reduce_sum(nonzero)
    size = tf.to_float(tf.size(layer))
    tf.scalar_summary(name, count/size)


def conv_1x1_layer(name, bottom, out_chans, activation, variables_to_restore, restore_sess, train):
    return conv_layer(name, bottom, 1, out_chans, activation, variables_to_restore, restore_sess, train)

def conv_3x3_layer(name, bottom, out_chans, activation, variables_to_restore, restore_sess, train):
    return conv_layer(name, bottom, 3, out_chans, activation, variables_to_restore, restore_sess, train)

def conv_layer(name, bottom, shape, out_chans, activation, variables_to_restore, restore_sess, train):
    in_chans = bottom.get_shape().as_list()[3]

    with tf.variable_scope(name) as scope:
        filter_shape = [shape, shape, in_chans, out_chans]

        filt = define_variable(variables_to_restore, restore_sess,  name + '/filter',
            tf.truncated_normal(filter_shape, stddev=0.01), name="filter",
            trainable=True)

        a = tf.nn.conv2d(bottom, filt, [1,1,1,1], padding="SAME")
        bn = batch_norm_2d(a, name, variables_to_restore, restore_sess, train)
        out = activation(bn)

        return out

def define_variable(variables_to_restore, restore_sess, restore_name, initial_value, trainable=True,
    collections=None, validate_shape=True, name=None):

    if restore_name in variables_to_restore:
        var_to_restore = variables_to_restore[restore_name]
        value = var_to_restore.eval(restore_sess)
        var = tf.constant(value, name=name) #maybe need dtype here
    else:
        var = tf.Variable(initial_value, trainable=trainable, collections=collections,
            validate_shape=validate_shape, name=name)
        variables_to_restore[restore_name] = var

    return var

# Returns three channel rgb image that is desaturated.
def desaturate(rgb):
    # take average value to desaturate.
    red, green, blue = tf.split(3, 3, rgb)
    return (red + green + blue) / 3

def batch_norm_2d(bottom, name, variables_to_restore, restore_sess, train):
    depth = bottom.get_shape().as_list()[-1]
    scale_after_norm = True
    epsilon = 0.001
    mean_name = name + "_bn/mean"
    variance_name = name + "_bn/variance"

    with tf.variable_scope(name + "_bn") as scope:
        beta = define_variable(variables_to_restore, restore_sess, name + "_bn/beta", tf.zeros([depth]), name="beta")
        gamma = define_variable(variables_to_restore, restore_sess, name + "_bn/gamma", tf.ones([depth]), name="gamma")

        if train:
            mean = tf.Variable(tf.zeros([depth]), trainable=False, name="mean")
            variance = tf.Variable(tf.ones([depth]), trainable=False, name="variance")

            ema = tf.train.ExponentialMovingAverage(decay=0.99)                  
            assigner = ema.apply([mean, variance])
            tf.add_to_collection("update_assignments", assigner)

            variables_to_restore[mean_name] = ema.average(mean)
            variables_to_restore[variance_name] = ema.average(variance)

            m, v = tf.nn.moments(bottom, [0, 1, 2])
            assign_mean = mean.assign(m)
            assign_variance = variance.assign(v)

            with tf.control_dependencies([assign_mean, assign_variance]):
              # maybe use ema.average(mean) here? 
              return tf.nn.batch_norm_with_global_normalization(
                  bottom, m, v, beta, gamma, epsilon, scale_after_norm)
        else:
            mean = define_variable(variables_to_restore, restore_sess, mean_name, tf.zeros([depth]), trainable=False, name="mean")
            variance = define_variable(variables_to_restore, restore_sess, variance_name, tf.ones([depth]), trainable=False, name="variance")

            return tf.nn.batch_norm_with_global_normalization(
                bottom, mean, variance, beta, gamma, epsilon, scale_after_norm)


def rgb2uv(rgb):
    yuv = rgb2yuv(rgb)
    _, u, v = tf.split(3, 3, yuv)
    return tf.concat(3, [u, v])

def dist(a, b):
    return tf.reduce_mean(tf.abs(a - b))

# eucledian distance 
def dist2(a, b):
    sq_diff = tf.square(a - b)
    sums = tf.reduce_sum(sq_diff, [1,2])
    dists = tf.sqrt(sums)
    return tf.reduce_mean(dists)

def blur_uv_loss(rgb, inferred_rgb):
    uv = rgb2uv(rgb)
    uv_blur0 = rgb2uv(blur(rgb, 3))
    uv_blur1 = rgb2uv(blur(rgb, 5))

    inferred_uv = rgb2uv(inferred_rgb)
    inferred_uv_blur0 = rgb2uv(blur(inferred_rgb, 3))
    inferred_uv_blur1 = rgb2uv(blur(inferred_rgb, 5))

    return ( dist(inferred_uv, uv) +
             dist(inferred_uv_blur0 , uv_blur0) +
             dist(inferred_uv_blur1, uv_blur1) ) / 3

def forward_model(grayscale, variables_to_restore, restore_sess):
    grayscale3 = tf.concat(3, [grayscale, grayscale, grayscale])
    _, gray_top_norm = vgg_top(grayscale3, variables_to_restore, restore_sess, train=False)
    inferred_color = color_encoder(gray_top_norm, variables_to_restore, restore_sess, train=False)
    inferred_rgb = rgb_inference(grayscale, inferred_color)
    return inferred_rgb


def training_model(rgb, variables_to_restore):
    grayscale = desaturate(rgb)
    grayscale3 = tf.concat(3, [grayscale, grayscale, grayscale])

    if FLAGS.hypercolumn_model:
        print "OLD hypercolumn MODEL"
        hypercolumns = hypercolumns_inference(grayscale3, variables_to_restore, None, train=True) 
        inferred_color = color_encoder_hypercolumns(hypercolumns, variables_to_restore, None, train=True)
    else:
        print "NEW MODEL"
        _, gray_top_norm = vgg_top(grayscale3, variables_to_restore, None, train=True)
        inferred_color = color_encoder(gray_top_norm, variables_to_restore, None, train=True)

    inferred_rgb = rgb_inference(grayscale, inferred_color)

    summary_image = build_summary_image(grayscale3, inferred_rgb, rgb, FLAGS.show_input)
    tf.image_summary("summary", summary_image, max_images=6)

    rgb_dist = dist(rgb, inferred_rgb)

    loss = blur_uv_loss(rgb, inferred_rgb)

    tf.scalar_summary("rgb_dist", rgb_dist)

    return loss, rgb_dist, summary_image

def define_avg(name, step_var, var):
    with tf.get_default_graph().device('/cpu:0'):
        history = tf.Variable(tf.zeros([NUM_AVG]), trainable=False)
        history_index = tf.mod(step_var, NUM_AVG)
        history_update = tf.scatter_update(history, history_index, var)
        tf.add_to_collection("checkpoint_vars", history)
        with tf.control_dependencies([history_update]):
            avg = tf.reduce_mean(history)
            tf.scalar_summary(name, avg)
            return avg

class TextData():
    def __init__(self, fn):
        self.fn = fn
        exists = os.path.isfile(fn)
        self.file = open(fn, 'ab')
        self.writer = csv.DictWriter(self.file, fieldnames=['ts', 'step', 'loss', 'rgb_dist'])
        if not exists:
            self.writer.writeheader()

    def write(self, step, loss, rgb_dist):
        t = datetime.datetime.now()
        ts = datetime.datetime.isoformat(t)

        self.writer.writerow({
            'ts': ts,
            'step': step,
            'loss': loss,
            'rgb_dist': rgb_dist,
        })
        self.file.flush()


def maybe_restore_checkpoint(sess, saver):
    if FLAGS.load:
        checkpoint = "20160103-imagenet/checkpoint-55685"
    else:
        checkpoint = tf.train.latest_checkpoint('.')

    if checkpoint:
        print "restoring from checkpoint", checkpoint
        saver.restore(sess, checkpoint)
    else:
        print "couldn't find checkpoint to restore from."

def print_variables_to_restore(variables_to_restore):
    for k in variables_to_restore:
        print k
        print variables_to_restore[k]
        print 

def run_training():
    print "==================\n================\nLooking for Directory: {}".format(FLAGS.train)
    data_set = data.DataSet(FLAGS.train)
    if data_set.length() < FLAGS.batch_size:
        FLAGS.batch_size = data_set.length()

    if not FLAGS.val:
        print "missing --val flag"
        exit(1)
    val_data_set = data.DataSet(FLAGS.val)

    rgb = tf.placeholder("float", [FLAGS.batch_size, 224, 224, 3], name="rgb")

    variables_to_restore = {}

    step_var = tf.Variable(0, trainable=False)
    variables_to_restore["step"] = step_var
    inc_step = step_var.assign_add(1) 

    loss, rgb_dist, summary_image = training_model(rgb, variables_to_restore)

    tf.scalar_summary("loss", loss)
    tf.scalar_summary("learning_rate", FLAGS.learning_rate)

    #avg_loss = define_avg("avg_loss", step_var, loss)
    #avg_rgb_dist = define_avg("avg_rgb_dist", step_var, rgb_dist)

    # Create a saver for writing training checkpoints.

    print_variables_to_restore(variables_to_restore)

    saver = tf.train.Saver(var_list=variables_to_restore, max_to_keep=3)

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter('log', flush_secs=5)

    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.0)
    train_op = opt.minimize(loss)

    # Important step for batch norm see batch_norm.py
    with tf.control_dependencies([train_op]):                                
        train_op2 = tf.group(*tf.get_collection("update_assignments"))

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    sess.run(tf.initialize_all_variables())

    train_csv = TextData('train-loss.csv')
    val_csv = TextData('val-loss.csv')

    maybe_restore_checkpoint(sess, saver)

    print "begin training"
    print "learning rate", FLAGS.learning_rate
    print "batch size", FLAGS.batch_size

    if FLAGS.fixed:
        feed_dict = { rgb: data_set.next_batch(FLAGS.batch_size) }

    while True:
        step = sess.run(inc_step)

        start_time = time.time()

        if not FLAGS.fixed:
            feed_dict = { rgb: data_set.next_batch(FLAGS.batch_size) }

        _, loss_value, rgb_dist_val = sess.run([train_op2, loss, rgb_dist], feed_dict=feed_dict)

        duration = time.time() - start_time

        # Print status to stdout.
        print('Step %d: rgb_dist = %.4f / loss = %.4f (%.1f sec)' % (step, rgb_dist_val, loss_value, duration))

        train_csv.write(step, loss_value, rgb_dist_val)

        if step % 50 == 0:
            # Update the events file.
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)

            # Save a checkpoint 
            saver.save(sess, './checkpoint', global_step=step)

        # now test the validation
        if step % 100 == 0:
            print "running validation..."
            count = 0
            summary_image_index = 0
            val_loss = []
            val_rgb_dist = []
            while count < 100:
                feed_dict = { rgb: val_data_set.next_batch(FLAGS.batch_size) }
                loss_value, rgb_dist_value, summary_image_value = sess.run([loss, rgb_dist, summary_image], feed_dict=feed_dict)

                for i in range(0, FLAGS.batch_size):
                    img = summary_image_value[i]
                    summary_image_index+=1
                    if summary_image_index <= 4:
                        fn = "val-imgs/val-%06d-%d.jpg" % (step, summary_image_index)
                        imsave(fn, img)

                val_loss.append(loss_value)
                val_rgb_dist.append(rgb_dist_value)
                count += FLAGS.batch_size

            val_avg_loss = np.mean(val_loss)
            val_avg_rgb_dist = np.mean(val_rgb_dist)

            print('VALIDATION SET: count = %d / rgb_dist = %.4f / loss = %.4f' % (count, val_avg_rgb_dist, val_avg_loss))

            val_csv.write(step, val_avg_loss, val_avg_rgb_dist)

def forward(fn):
    img = data.load_image(fn)
    gray = (img[:,:,0] + img[:, :, 1] + img[:, :, 2]) / 3.0
    batch = gray.reshape((1, 224, 224, 1))

    variables_to_restore = {}
    grayscale = tf.placeholder("float", [1, 224, 224, 1], name="grayscale")
    inferred_rgb = forward_model(grayscale, variables_to_restore, None)

    print_variables_to_restore(variables_to_restore)

    saver = tf.train.Saver(var_list=variables_to_restore)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    checkpoint = tf.train.latest_checkpoint('.')
    assert checkpoint
    print "restoring from checkpoint", checkpoint
    saver.restore(sess, checkpoint)

    colorized = sess.run(inferred_rgb, feed_dict={ grayscale: batch })
    assert colorized.shape == (1, 224, 224, 3)
    imsave("out.jpg", colorized[0])
    print "saved to out.jpg"

    if FLAGS.save:
        construct_const_graph(variables_to_restore, sess)

def construct_const_graph(variables_to_restore, restore_sess):
    const_graph = tf.Graph()

    with const_graph.as_default():
        grayscale = tf.placeholder("float", [1, 224, 224, 1], name="grayscale")
        inferred_rgb = forward_model(grayscale, variables_to_restore, restore_sess)

    graph_def = const_graph.as_graph_def()
    print "graph_def byte size", graph_def.ByteSize()
    graph_def_s = graph_def.SerializeToString()

    save_path = "colorize.tfmodel"
    with open(save_path, "wb") as f:
      f.write(graph_def_s)

    print "saved model to %s" % save_path




def main(_):
    if FLAGS.forward:
        forward(FLAGS.forward)
    else:
        #with tf.get_default_graph().device('/cpu:0'):
        run_training()

if __name__ == '__main__':
    tf.app.run()
