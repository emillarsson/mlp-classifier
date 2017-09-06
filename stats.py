import tensorflow as tf
import numpy as np

from collections import defaultdict
from tensorflow.contrib.tensorboard.plugins import projector


class Stats(object):
    def __init__(self, sess, summary_writer, scalar_tags=None, histogram_tags=None):
        self.sess = sess
        self.writer = summary_writer
        self.scalars = scalar_tags is not None
        self.histogram = histogram_tags is not None

        if self.scalars:
            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])

        if self.histogram:
            self.histogram_placeholders = {}
            self.histogram_ops = {}
            self.histogram_step = 100
            self.reset_histogram_inputs()

            for tag in histogram_tags:
                self.histogram_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.histogram_ops[tag] = tf.summary.histogram(tag, self.histogram_placeholders[tag])

      
    def reset_histogram_inputs(self):
        self.histogram_inputs = {}

    def update(self, scalar_dict, histogram_dict, step):
        if self.scalars:
            self.inject_summary(scalar_dict, step)

        if self.histogram:
            for tag in histogram_dict:
                self.histogram_inputs[tag] = []
                self.histogram_inputs[tag].extend(histogram_dict[tag])

            if step % self.histogram_step:
                self.inject_histogram(histogram_dict, step)
                self.reset_histogram_inputs()


    def inject_summary(self, dictionary, step):
        summary_list = self.sess.run([self.summary_ops[tag] for tag in dictionary.keys()],
            {self.summary_placeholders[tag]: value for tag, value in dictionary.items()})
        for summary in summary_list:
            self.writer.add_summary(summary, step)
            self.writer.flush()

    def inject_histogram(self, dictionary, step):
        histogram_lists = self.sess.run([self.histogram_ops[tag] for tag in dictionary.keys()],
            {self.histogram_placeholders[tag]: value for tag, value in dictionary.items()})
        for histogram in histogram_lists:
            self.writer.add_summary(histogram, step)
            self.writer.flush()
