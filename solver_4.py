import time
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization


TF_DTYPE = tf.float32

class FeedForwardModel(object):

    def __init__(self, sess, bsde, config):
        self._sess = sess
        self.config = config
        self.bsde = bsde
        
        self.dim = bsde._dim
        self.total_time = bsde._total_time
        self.num_time_interval = bsde._num_time_interval
        self.delta_t = bsde._delta_t
                
        self.f_network = []
        self.z_network=[]
    
    
    def train(self):
        self.start_time = time.time()
    
        dw_valid,x_valid = self.bsde.sample(self.config.valid_size)
        feed_dict_valid = {self._x: x_valid,self._dw:dw_valid,
                           self.lambda1:1,self.lambda2:0, 
                           self._is_training: False}
        
        loss = 1000
        counter = 1
        
#         while loss > 10:
#             print('the '+str(counter)+' time of pre-train')
#             self._sess.run(tf.global_variables_initializer())
#             for step in range(self.config.pre_train_num_iteration+1):
#                 if step % self.config.logging_frequency == 0:
#                     loss= self._sess.run(self._loss, feed_dict = feed_dict_valid)
#                     elapsed_time = time.time() - self.start_time + self._t_build
#                     print("step: %5u,loss: %.4e,  elapsed time %3u" % (step, loss, elapsed_time))
#                 dw_train, x_train = self.bsde.sample(self.config.batch_size)
#                 loss = self._sess.run([self._loss ,self._train_ops], 
#                                     feed_dict={self._x: x_train, self._dw:dw_train, self.lambda1:1,
#                                                self.lambda2:0, self._is_training: True})[0]
#             counter+=1
#         print('Finish pre train')
        
        feed_dict_valid = {self._x: x_valid, self._dw:dw_valid, 
                           self.lambda1: 0, self.lambda2: 1, self._is_training: False}
        self._sess.run(tf.global_variables_initializer())
        
        for step in range(self.config.num_iterations + 1):
            if step % self.config.logging_frequency == 0:
                loss, init = self._sess.run([self._loss, self._y_init], feed_dict=feed_dict_valid)
                elapsed_time = time.time() - self.start_time + self._t_build
                print("step: %5u,loss: %.4e,y_init: %.4e  elapsed time %3u" % (step, loss, init, elapsed_time))
            dw_train,x_train = self.bsde.sample(self.config.batch_size)
            loss=self._sess.run([self._loss ,self._train_ops], 
                                feed_dict={self._x: x_train,self._dw:dw_train,self.lambda1:0,
                                           self.lambda2:1,self._is_training: True})[0]
   
    
    def nn_structure(self, _network, _input):
        z = _network[0](_input)
        for i in range(1, len(_network)):
            z = _network[i](z)
        return z
    
    def relu_adding(self, network, unit_lst, activation = None):
        #num >=2
        num = len(unit_lst)
        network.append(Dense(units = unit_lst[0], input_shape = (self.dim,), activation = 'relu'))
        for i in range(1, num):
            network.append(Dense(units = unit_lst[i], input_shape = (unit_lst[i-1],), activation = 'relu'))
        network.append(Dense(units = self.dim, input_shape = (unit_lst[-1],), activation = activation))
    
    def build(self):
        start_time = time.time()
        time_stamp = np.arange(0, self.num_time_interval) * self.delta_t
        
     #   self.relu_adding(self.f_network, self.config.f_units, 'relu')
        
        for i in range(self.num_time_interval):
            temp = []
            temp.append(BatchNormalization())
            self.relu_adding(temp, self.config.z_units)
            self.z_network.append(temp) 
        
        self._x = tf.placeholder(TF_DTYPE, [None,self.dim, self.num_time_interval+1], name='X')
        self._dw = tf.placeholder(TF_DTYPE, [None, self.dim, self.num_time_interval], name='dW')
        self.lambda1=tf.placeholder(TF_DTYPE, name='lambda1')
        self.lambda2=tf.placeholder(TF_DTYPE, name='lambda2')
        self._is_training = tf.placeholder(tf.bool)
        
        self._y_init = tf.Variable(tf.random_uniform([1],
                                                     minval=self.config.y_init_range[0],
                                                     maxval=self.config.y_init_range[1],
                                                     dtype=TF_DTYPE))

#         x = np.linspace(self.bsde.lower, self.bsde.upper, self.config.ob_num)
#         self.x = tf.constant(x, dtype=TF_DTYPE, shape=[1, self.config.ob_num, 1])

        with tf.variable_scope('forward'):
            z = self.nn_structure(self.z_network[0], self._x[:,:,0])
            
            all_one_vec = tf.ones(shape=tf.stack([tf.shape(self._dw)[0], 1]), dtype=TF_DTYPE)
            y = all_one_vec * self._y_init
            
            for t in range(0, self.num_time_interval - 1):
                y = y - self.delta_t * (
                    self.bsde.f_tf(time_stamp[t], self._x[:, :, t], y, z))
                
                y = y + tf.reduce_sum(z * self.bsde._sigma * self._x[:, :, t]
                * self._dw[:, :, t], 1, keepdims=True)
                
                z = self.nn_structure(self.z_network[ t+1 ], self._x[:, :, t + 1])
                       
            

            y = y - self.bsde.delta_t * self.bsde.f_tf(
                time_stamp[-1], self._x[:, :, -2], y, z)
            y = y + tf.reduce_sum(z * self.bsde._sigma *self._x[:, :, -2]
                * self._dw[:, :, -1], 1, keepdims=True)
            
         #   loss1 = y_init - self.bsde.g_tf(0, self._x[:, :, 0])                  
            loss2 = y - self.bsde.g_tf(self.total_time, self._x[:, :, -1])
            
            self._loss = self.lambda2 * tf.reduce_mean(tf.square(loss2))
#            
#            self.f_graphs = []
#            
#             l = self.nn_structure(self.f_network, self.x + 0.0)
#             self.f_graphs = l
            
#             self.z_graphs=[]
#             for t in range(self.num_time_interval):
#                 l = self.nn_structure(self.z_network[t], self.x[0])
#                 self.z_graphs.append(l)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self._loss, trainable_variables)
        
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False, dtype=tf.int32)
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    self.config.lr_boundaries,
                                                    self.config.lr_values)
        
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),name='train_step')
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=global_step,name='train_step')
        all_ops = [apply_op] #+ self._extra_train_ops
        self._train_ops = tf.group(*all_ops)
        self._t_build = time.time()-start_time



