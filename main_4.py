import matplotlib.pyplot as plt
from equation import PricingOption, AllenCahn, HJB
from config import PricingOptionConfig, AllenCahnConfig, HJBConfig
from solver_4 import FeedForwardModel
import tensorflow.compat.v1 as tf
import numpy as np
from scipy.stats import norm

# 4.3
#params
dim, total_time, num_time_interval = 100, 1, 10

#fit
H = HJB(dim, total_time, num_time_interval)
tf.reset_default_graph()
with tf.Session() as sess:
    model = FeedForwardModel(sess, H, HJBConfig())
    model.build()
    model.train()
    
# 4.2
#params
dim, total_time, num_time_interval = 100, 0.3, 10

#fit
AC= AllenCahn(dim, total_time, num_time_interval)
tf.reset_default_graph()
with tf.Session() as sess:
    model = FeedForwardModel(sess, AC, AllenCahnConfig())
    model.build()
    model.train()
    
# 4.4
#params
dim, total_time, num_time_interval = 100, 0.5, 10

#fit
Option= PricingOption(dim, total_time, num_time_interval)
tf.reset_default_graph()
with tf.Session() as sess:
    model = FeedForwardModel(sess, Option, PricingOptionConfig())
    model.build()
    model.train()
