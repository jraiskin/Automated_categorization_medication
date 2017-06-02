try:
    from utils.utils import seed
    from utils.utils_nn import hard_tanh
except:
    from utils import seed
    from utils_nn import hard_tanh

import numpy as np
np.random.seed(seed())

import random
random.seed(seed())

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.tensorboard.plugins import projector  # embeddings visualizer

import os

class Rnn_model(object):

    def __init__(self, 
                 *args, 
                 hparam_str=None, 
                 seq_len, 
                 n_class, 
                 n_char, 
                 char_embed_dim, 
                 one_hot, 
                 hidden_state_size, 
                 keep_prob, 
                 learn_rate, 
                 dynamic_learn_rate, 
                 rnn_type, 
                 bidirection, 
                 top_k, 
                 epochs, 
                 log_dir, 
                 embed_vis_path=None, 
                 summary_step, 
                 save_step, 
                 seed, 
                 activation_function, 
                 learn_p_delta_scale, 
                 noise_act_alpha, 
                 noise_act_half_normal, 
                 l2_weight_reg, 
                 target_rep, 
                 target_rep_weight, 
                 verbose_summary, 
                 feed_dict_train=None, 
                 feed_dict_test=None, 
                 **kwargs):
        
        self.hparam_str = hparam_str
        self.seq_len = seq_len 
        self.n_class = n_class 
        self.n_char = n_char
        self.char_embed_dim = char_embed_dim
        self.one_hot = one_hot
        self.hidden_state_size = hidden_state_size
        self.learn_rate = learn_rate
        self.dynamic_learn_rate = dynamic_learn_rate
        self.rnn_type = rnn_type
        self.bidirection = bidirection
        self.top_k = top_k
        self.epochs = epochs
        self.log_dir = log_dir
        self.embed_vis_path = embed_vis_path
        self.summary_step = summary_step 
        self.save_step = save_step
        self.seed = seed
        self.activation_function = activation_function 
        self.learn_p_delta_scale = learn_p_delta_scale 
        self.noise_act_alpha = noise_act_alpha
        self.noise_act_half_normal = noise_act_half_normal
        self.l2_weight_reg = l2_weight_reg
        self.target_rep = target_rep
        self.verbose_summary = verbose_summary
        self.target_rep_weight = target_rep_weight if self.target_rep else 0.0
        self.embedding_matrix = None

        # clear tf graph and set seeds
        tf.reset_default_graph()
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # Setup placeholders, and reshape the data
        self.x_ = tf.placeholder(tf.int32, [None, self.seq_len], 
                            name='Examples')
        self.y_ = tf.placeholder(tf.int32, [None, self.n_class], 
                            name='Lables')
        self.keep_prob = tf.placeholder(tf.float32, [], 
                            name='Keep_probability')
        self.use_noise = tf.placeholder(tf.bool, [], 
                            name='Use_noise')
        # indicator that there will be no training
        self.no_train = self.hparam_str is None and self.embed_vis_path is None
        # set hyper-parameter p,
        # as part of noise scaling in the noisy activation function
        if self.learn_p_delta_scale and self.activation_function == 'noisy_tanh':
            self.p_delta_scale = tf.Variable(
                tf.truncated_normal([self.hidden_state_size], 
                                    stddev=0.1, 
                                    seed=self.seed), 
                name='p_delta_scale')
        else:
            self.p_delta_scale = 1.0
        
        # set activation function
        if hasattr(self.activation_function, '__call__'):
            pass
        elif self.activation_function == 'noisy_tanh':
            self.activation_function = self.noise_tanh_p
        elif self.activation_function == 'tf.tanh':
            self.activation_function = tf.tanh
        else:
            raise ValueError('Received an unknown activation function')
        
        if feed_dict_train is not None:
            self.feed_dict_train = {self.x_: feed_dict_train['x'], 
                                    self.y_: feed_dict_train['y'], 
                                    self.keep_prob: keep_prob, 
                                    self.use_noise: True}

            self.feed_dict_train_eval = {**self.feed_dict_train, 
                                         **{self.keep_prob: 1.0, 
                                            self.use_noise: False}}
        
        if feed_dict_test is not None:
            self.feed_dict_test = {self.x_: feed_dict_test['x'], 
                                   self.y_: feed_dict_test['y'], 
                                   self.keep_prob: 1.0, 
                                   self.use_noise: False}

        self.embedding_matrix = self.embed_matrix()

        self.outputs = self.rnn_unit(input=self.x_)
        with tf.name_scope('logits_seq'):
            if self.bidirection: logit_in_size = 2 * self.hidden_state_size
            else: logit_in_size = self.hidden_state_size
            self.logits = [self.logit(input=out, 
                                      size_in=logit_in_size, 
                                      size_out=self.n_class) 
                           for out in self.outputs]

        with tf.name_scope('Cost_function'):
            """
            Cross entropy loss with target replication and
            regularization terms based on the weights' L2 norm
            """
            with tf.name_scope('target_replication_loss'):
                self.cost_targetrep = tf.reduce_mean(
                    [tf.nn.softmax_cross_entropy_with_logits(
                        logits=log, labels=self.y_) 
                     for log in self.logits])
            with tf.name_scope('cross_entropy'):
                self.cost_crossent = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.logits[-1], labels=self.y_))
            with tf.name_scope('L2_norm_reg'):
                self.cost_l2reg = tf.reduce_mean([tf.nn.l2_loss(weight) 
                                                  for weight in tf.trainable_variables()])
            with tf.name_scope('total_cost'):
                self.cost = self.target_rep_weight * self.cost_targetrep + \
                    (1 - self.target_rep_weight) * self.cost_crossent + \
                    self.l2_weight_reg * self.cost_l2reg
            # add summaries
            tf.summary.scalar('Total_cost_train', 
                              self.cost, collections=['train'])
            tf.summary.scalar('Total_cost_test', 
                              self.cost, collections=['test'])
            
        with tf.name_scope('Cost_function_additional_metrics'):
            tf.summary.scalar('Target_rep_cost_train', 
                              self.cost_targetrep, collections=['train'])
            tf.summary.scalar('Target_rep_cost_test', 
                              self.cost_targetrep, collections=['test'])
            tf.summary.scalar('Cross_entropy_train', 
                              self.cost_crossent, collections=['train'])
            tf.summary.scalar('Cross_entropy_test', 
                              self.cost_crossent, collections=['test'])
            tf.summary.scalar('L2_norm_train', 
                              self.cost_l2reg, collections=['train'])
            tf.summary.scalar('L2_norm_test', 
                              self.cost_l2reg, collections=['test'])            
            
        with tf.name_scope('Train'):
            if self.dynamic_learn_rate:
                self.optimizer = tf.train.GradientDescentOptimizer(self.learn_rate)
            else:
                self.optimizer = tf.train.AdamOptimizer(self.learn_rate)
            self.train_step = self.optimizer.minimize(self.cost)

        with tf.name_scope('Accuracy'):  # takes the last element of logits
            self.correct_prediction = tf.equal(tf.argmax(self.logits[-1], 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            tf.summary.scalar('Accuracy_train', self.accuracy, collections=['train'])
            tf.summary.scalar('Accuracy_test', self.accuracy, collections=['test'])
        
        with tf.name_scope('Mean_Reciprocal_Rank'):  # takes the last element of logits
            self.recip_rank = tf.reduce_mean(
                self.get_reciprocal_rank(self.logits[-1], 
                                         self.y_, 
                                         True))
            tf.summary.scalar('Mean_Reciprocal_Rank_train', 
                              self.recip_rank, collections=['train'])
            tf.summary.scalar('Mean_Reciprocal_Rank_test', 
                              self.recip_rank, collections=['test'])        
        
        with tf.name_scope('In_top_{}'.format(self.top_k)):  # takes the last element of logits
            self.y_targets = tf.argmax(self.y_, 1)
            self.top_k_res = tf.reduce_mean(tf.cast(
                tf.nn.in_top_k(self.logits[-1], self.y_targets, self.top_k), 
                tf.float32))
            tf.summary.scalar('In_top_{}_train'.format(self.top_k), self.top_k_res, collections=['train'])
            tf.summary.scalar('In_top_{}_test'.format(self.top_k), self.top_k_res, collections=['test'])

        # summaries per collection and saver object
        self.summ_train = tf.summary.merge_all('train')
        self.summ_test = tf.summary.merge_all('test')
        self.saver = tf.train.Saver()
        self.init_op = tf.global_variables_initializer()
        
        # init vars and setup writer
        self.sess = tf.Session()
        self.sess.run(self.init_op)
        if not self.no_train:
            self.writer = tf.summary.FileWriter(self.log_dir + self.hparam_str)
            self.writer.add_graph(self.sess.graph)

            # Add embedding tensorboard visualization. Need tensorflow version
            self.config = projector.ProjectorConfig()
            self.embed = self.config.embeddings.add()
            self.embed.tensor_name = self.embedding_matrix.name
            self.embed.metadata_path = os.path.join(self.embed_vis_path)
            projector.visualize_embeddings(self.writer, self.config)
        
        
    def embed_matrix(self, stddev=0.1, name='embeddings'):
        # index_size would be the size of the character set
        with tf.name_scope(name):
            if not self.one_hot:
                embedding_matrix = tf.get_variable(
                    'embedding_matrix', 
                    initializer=tf.truncated_normal([self.n_char, self.char_embed_dim], 
                                                    stddev=stddev, 
                                                    seed=self.seed), 
                    trainable=True)
            else:
                # creating a one-hot for each character corresponds to the identity matrix
                embedding_matrix = tf.constant(value=np.identity(self.n_char), 
                                               name='embedding_matrix', 
                                               dtype=tf.float32)
                self.char_embed_dim = self.n_char
            if self.verbose_summary:
                tf.summary.histogram('embedding_matrix', embedding_matrix, collections=['train'])
            self.embedding_matrix = embedding_matrix
            return self.embedding_matrix
        
        
    def rnn_unit(self, 
                  input, 
                  name='LSTM'):
        # check, then set the right name
        assert self.rnn_type in ['LSTM', 'GRU'], \
            'rnn_type has to be either LSTM or GRU'
        name = 'LSTM' if self.rnn_type == 'LSTM' else 'GRU'
        if self.bidirection: name += '_bidir'
        with tf.name_scope(name):
            input = tf.nn.embedding_lookup(self.embedding_matrix, input)
            # reshaping
            # Permuting batch_size and n_steps
            input = tf.transpose(input, [1, 0, 2])
            # Reshaping to (n_steps*batch_size, n_input)
            input = tf.reshape(input, [-1, self.char_embed_dim])
            # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            rnn_inputs = tf.split(input, self.seq_len, 0)
            
            # setting the correct RNN cell type (LSTM of GRU)
            rnn_cell = rnn.BasicLSTMCell if self.rnn_type == 'LSTM' \
                else rnn.GRUCell
            # setting the args (forget_bias applies only to LSTM)
            rnn_cell_args = {'num_units': self.hidden_state_size, 
                             'activation': self.activation_function}
            
            if 'LSTMCell' in str(rnn_cell.__call__ ):
                rnn_cell_args['forget_bias'] = 1.0
            rnn_cell(**rnn_cell_args)
            
            cell_fw = rnn_cell(**rnn_cell_args)
            cell_fw = rnn.DropoutWrapper(cell_fw, 
                                         output_keep_prob=self.keep_prob, 
                                         seed=self.seed)
            
            if self.bidirection:
                # add another cell for backwards direction and a dropout wrapper
                cell_bw = rnn_cell(**rnn_cell_args)
                cell_bw = rnn.DropoutWrapper(cell_bw, 
                                             output_keep_prob=self.keep_prob, 
                                             seed=self.seed)
                outputs, _, _ = rnn.static_bidirectional_rnn(
                    cell_fw, cell_bw, rnn_inputs, dtype=tf.float32, scope=name)
            else:
                outputs, _ = rnn.static_rnn(cell_fw, rnn_inputs, dtype=tf.float32, scope=name)
            
            if not self.target_rep:  # take only last output (list for structure consistency)
                outputs = [outputs[-1]]
            if self.verbose_summary:
                tf.summary.histogram('outputs', outputs, collections=['train'])
            return outputs


    def logit(self, 
              input, 
              size_in, 
              size_out, 
              stddev=0.1, 
              name='logit'):

        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([size_in, size_out], 
                                                stddev=stddev, 
                                                seed=self.seed), 
                            name='W')
            b = tf.Variable(tf.constant(0.1, 
                                        shape=[size_out]), 
                            name='B')
            logits = tf.matmul(input, w) + b
            if self.verbose_summary:
                tf.summary.histogram('weights', w, collections=['train'])
                tf.summary.histogram('biases', b, collections=['train'])
                tf.summary.histogram('logits', logits, collections=['train'])
            return logits
    
        
    def train(self):
        print('Starting to train model {:s}'.format(self.hparam_str))
        for i in range(1, self.epochs+1):
            # update learning rate, if it is dynamic
            if self.dynamic_learn_rate: self.update_lr(epoch=i)
            # train step
            self.sess.run(self.train_step, feed_dict=self.feed_dict_train)
            if i % self.summary_step == 0:
                # train summary
                # use self.feed_dict_train_eval for evaluation (keep probability set to 1.0)
                [train_accuracy, train_cost, _, train_top_k] = \
                    self.run_eval(feed_dict=self.feed_dict_train_eval,
                                  step=i, 
                                  summary=self.summ_train)
                print('{:.3f} of observations in the top is {}'.format(train_top_k, self.top_k))
                # test summary
                [test_accuracy, test_cost, _, test_top_k] = \
                    self.run_eval(feed_dict=self.feed_dict_test,
                                  step=i, 
                                  summary=self.summ_test)
                print('Epoch number {}, '.format(i) +
                      'training accuracy is {:.5f} and '.format(train_accuracy) + 
                      'test accuracy is {:.5f}, '.format(test_accuracy))
                print('training cost is {:.5f} and '.format(train_cost) + 
                      'test cost is {:.5f}'.format(test_cost))
                
            if i % self.save_step == 0:
                print('Saving step {}'.format(i))
                self.saver.save(self.sess, os.path.join(self.log_dir, 
                                                        self.hparam_str, 
                                                        'model.ckpt'), i)
            
        print('Training the model is done! ({:s})'.format(self.hparam_str))
        

    def run_eval(self, *, feed_dict=None, 
                 step=None, summary=None, session=None):
        """
        Run all evaluation metrics with the given feed_dict.
        Use summary (optional) to specify a summary group, 
        by providing the merged summary object.
        session defaults to the internal session (self.sess).
        Returns [accuracy, cost, recip_rank, top_k]
        """
        assert not (summary is not None and step is None), \
            'If summary is chosen, a step must be specified.'
        
        if session is None:
            session = self.sess
        
        if feed_dict is None:
            feed_dict = self.feed_dict_test
        
        if summary is not None:
            [accuracy, cost,_ , _, _, recip_rank, top_k, s] = \
                session.run([self.accuracy, 
                             self.cost, 
                             self.cost_targetrep, 
                             self.cost_crossent, 
                             self.cost_l2reg, 
                             self.recip_rank, 
                             self.top_k_res, 
                             summary],
                            feed_dict=feed_dict)
            self.writer.add_summary(s, step)
        else:
            [accuracy, cost,_ , _, _, recip_rank, top_k] = \
                session.run([self.accuracy, 
                             self.cost, 
                             self.cost_targetrep, 
                             self.cost_crossent, 
                             self.cost_l2reg, 
                             self.recip_rank, 
                             self.top_k_res],
                            feed_dict=feed_dict)

        return [accuracy, cost, recip_rank, top_k]
    
    
    def update_test_dict(self, feed_dict_test):
        self.feed_dict_test = {self.x_: feed_dict_test['x'], 
                               self.y_: feed_dict_test['y'], 
                               self.keep_prob: 1.0, 
                               self.use_noise: False}
    
    
    def noise_tanh_p(self,
                     x,
                     p=None,
                     use_noise=None,
                     alpha=None,
                     c=0.5,
    #                  noise=None,
                     half_normal=None):
        """
        Noisy Hard Tanh Units: NAN with learning p
        https://arxiv.org/abs/1603.00391
        Arguments:
            x: input tensor variable.
            p: tensorflow variable, a vector of parameters for p
            use_noise: bool, whether to add noise or not (useful for test time)
            c: float, standard deviation of the noise
            alpha: float, the leakage rate from the linearized function to the clipped function.
            half_normal: bool, whether the noise should be sampled from half-normal or
            normal distribution.
        """
        if p is None:
            p=self.p_delta_scale
        if use_noise is None:
            use_noise=self.use_noise
        if alpha is None:
            alpha = self.noise_act_alpha
        if half_normal is None:
            half_normal = self.noise_act_half_normal
        
        signs = tf.sign(x)
    #     delta = HardTanh(x) - x
        delta = x - hard_tanh(x)

        scale = c * (0.5 - tf.sigmoid(p * delta))**2
#         scale = c * (0.5 - tf.sigmoid(delta))**2
        if alpha > 1.0 and half_normal:
               scale *= -1.0

        zeros = tf.zeros(tf.shape(x), dtype=tf.float32, name=None)
        rn_noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)
    #     def noise_func() :return tf.abs(rn_noise) if half_normal else zeros
        def noise_func() :return tf.abs(rn_noise) if half_normal else rn_noise
        def zero_func (): return zeros + 0.7979 if half_normal else zeros
        noise = tf.cond(use_noise,noise_func,zero_func)
        
        res = alpha * hard_tanh(x) + (1.0 - alpha) * x - signs * scale * noise
        return res
    
    
    def tf_get_rank_order(self, input, reciprocal):
        """
        Returns a tensor of the rank of the input tensor's elements.
        rank(highest element) = 1.
        """
        assert isinstance(reciprocal, bool), 'reciprocal has to be bool'
        size = tf.size(input)
        indices_of_ranks = tf.nn.top_k(-input, k=size)[1]
        indices_of_ranks = size - tf.nn.top_k(-indices_of_ranks, k=size)[1]
        if reciprocal:
            indices_of_ranks = tf.cast(indices_of_ranks, tf.float32)
            indices_of_ranks = tf.map_fn(
                lambda x: tf.reciprocal(x), indices_of_ranks, 
                dtype=tf.float32)
            return indices_of_ranks
        else:
            return indices_of_ranks
    
    
    def get_reciprocal_rank(self, logits, targets, reciprocal=True):
        """
        Returns a tensor containing the (reciprocal) ranks
        of the logits tensor (wrt the targets tensor).
        The targets tensor should be a 'one hot' vector 
        (otherwise apply one_hot on targets, such that index_mask is a one_hot).
        """
        function_to_map = lambda x: self.tf_get_rank_order(x, reciprocal=reciprocal)
        ordered_array_dtype = tf.float32 if reciprocal is not None else tf.int32
        ordered_array = tf.map_fn(function_to_map, logits, 
                                  dtype=ordered_array_dtype)

        size = int(logits.shape[1])
        index_mask = tf.reshape(
                targets, [-1,size])
        if reciprocal:
            index_mask = tf.cast(index_mask, tf.float32)

        return tf.reduce_sum(ordered_array * index_mask,1)
    
    
    def restore(self, cp_path, feed_dict = None):
        
        print('Loading variables from {:s}'.format(cp_path))

        ckpt = tf.train.get_checkpoint_state(cp_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("no checkpoint found")
        
        print('Loading successful')
    
    def close_session(self):
        self.sess.close()
    
    def update_lr(self, epoch):
        self.learn_rate = 1.0 / np.sqrt(epoch)
