from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import helper_tf_util
import time
import math
from utils.sampling import tf_sampling


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


def sampling(batch_size, npoint, pts, feature=None):
    '''
    inputs:
    npoint: scalar, number of points to sample
    pointcloud: B * N * D, input point cloud
    output:
    sub_pts: B * npoint * D, sub-sampled point cloud
    '''
    fps_idx = tf_sampling.farthest_point_sample(npoint, pts)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1)), (1, npoint,1))
    idx = tf.concat([batch_indices, tf.expand_dims(fps_idx, axis=2)], axis=2)
    idx.set_shape([batch_size, npoint, 2])
    if feature is None:
        return tf.gather_nd(pts, idx)
    else:
        return tf.gather_nd(pts, idx), tf.gather_nd(feature, idx)


class Network:
    def __init__(self, dataset, config):
        flat_inputs = dataset.flat_inputs
        self.config = config
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                self.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            num_layers = self.config.num_layers
            self.inputs['features'] = flat_inputs[0]
            self.inputs['labels'] = flat_inputs[1]
            self.inputs['input_inds'] = flat_inputs[2]
            self.inputs['cloud_inds'] = flat_inputs[3]

            self.labels = self.inputs['labels']
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            self.class_weights = DP.get_class_weights(dataset.name)
            self.time_stamp = time.strftime('_%Y-%m-%d_%H-%M-%S', time.gmtime())
            self.Log_file = open('log_train_' + dataset.name + str(dataset.val_split) + self.time_stamp + '.txt', 'a')

        with tf.variable_scope('layers'):
            self.logits, self.new_xyz, self.xyz = self.inference(self.inputs, self.is_training)

        #####################################################################
        # Ignore the invalid point (unlabeled) when calculating the loss #
        #####################################################################
        with tf.variable_scope('loss'):
            self.logits = tf.reshape(self.logits, [-1, config.num_classes])
            self.labels = tf.reshape(self.labels, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)

            aug_loss_weights = tf.constant([0.1, 0.1, 0.3, 0.5, 0.5])
            aug_loss = 0
            for i in range(self.config.num_layers):
                centroids = tf.reduce_mean(self.new_xyz[i], axis=2)
                relative_dis = tf.sqrt(tf.reduce_sum(tf.square(centroids-self.xyz[i]), axis=-1) + 1e-12)
                aug_loss = aug_loss + aug_loss_weights[i] * tf.reduce_mean(tf.reduce_mean(relative_dis, axis=-1), axis=-1)

            self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights) + aug_loss

        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'):
            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def inference(self, inputs, is_training):

        d_out = self.config.d_out
        ratio = self.config.sub_sampling_ratio
        k_n = self.config.k_n
        feature = inputs['features']
        og_xyz = feature[:, :, :3] 
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)

        # ###########################Encoder############################
        f_encoder_list = []
        input_xyz = og_xyz
        input_up_samples = []
        new_xyz_list = []
        xyz_list = []
        n_pts = self.config.num_points
        for i in range(self.config.num_layers):
            # Farthest Point Sampling:
            input_neigh_idx = tf.py_func(DP.knn_search, [input_xyz, input_xyz, k_n], tf.int32)
            n_pts = n_pts // ratio[i]
            sub_xyz, inputs_sub_idx = tf.cond(tf.equal(is_training, tf.constant(True)), lambda: sampling(self.config.batch_size, n_pts, input_xyz, input_neigh_idx), lambda: sampling(self.config.val_batch_size, n_pts, input_xyz, input_neigh_idx))
            inputs_interp_idx = tf.py_func(DP.knn_search, [sub_xyz, input_xyz, 1], tf.int32)
            input_up_samples.append(inputs_interp_idx)

            # Bilateral Context Encoding
            f_encoder_i, new_xyz = self.bilateral_context_block(feature, input_xyz, input_neigh_idx, d_out[i],
                                                 'Encoder_layer_' + str(i), is_training)
            f_sampled_i = self.random_sample(f_encoder_i, inputs_sub_idx)
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
            xyz_list.append(input_xyz)
            new_xyz_list.append(new_xyz)
            input_xyz = sub_xyz
        # ###########################Encoder############################
        
        # ###########################Decoder############################
        # Adaptive Fusion Module
        f_multi_decoder = [] # full-sized feature maps
        f_weights_decoders = [] # point-wise adaptive fusion weights
        for n in range(self.config.num_layers):
            feature = f_encoder_list[-1-n]
            feature = helper_tf_util.conv2d(feature, feature.get_shape()[3].value, [1, 1],
                                        'decoder_0' + str(n),
                                        [1, 1], 'VALID', True, is_training)
            f_decoder_list = []
            for j in range(self.config.num_layers-n):
                f_interp_i = self.nearest_interpolation(feature, input_up_samples[-j - 1 -n])
                f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([f_encoder_list[-j - 2 -n], f_interp_i], axis=3),
                                                              f_encoder_list[-j - 2 -n].get_shape()[-1].value, [1, 1],
                                                              'Decoder_layer_' + str(n) + '_' + str(j), [1, 1], 'VALID', bn=True,
                                                              is_training=is_training)
                feature = f_decoder_i
                f_decoder_list.append(f_decoder_i)
            # collect full-sized feature maps which are upsampled from multiple resolutions
            f_multi_decoder.append(f_decoder_list[-1])
            # summarize point-level information
            curr_weight = helper_tf_util.conv2d(f_decoder_list[-1], 1, [1, 1], 'Decoder_weight_' + str(n), [1, 1], 'VALID', bn=False, activation_fn=None)
            f_weights_decoders.append(curr_weight)
        # regress the fusion parameters
        f_weights = tf.concat(f_weights_decoders, axis=-1)
        f_weights = tf.nn.softmax(f_weights, axis=-1)
        # adptively fuse them by calculating a weighted sum
        f_decoder_final = tf.zeros_like(f_multi_decoder[-1])
        for i in range(len(f_multi_decoder)):
            f_decoder_final = f_decoder_final + tf.tile(tf.expand_dims(f_weights[:,:,:,i], axis=-1), [1, 1, 1, f_multi_decoder[i].get_shape()[-1].value]) * f_multi_decoder[i]
        # ###########################Decoder############################

        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_final, 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)     
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        f_out = tf.squeeze(f_layer_fc3, [2])
        return f_out, new_xyz_list, xyz_list

    def train(self, dataset):
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        self.sess.run(dataset.train_init_op)
        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits,
                       self.labels,
                       self.accuracy]
                _, _, summary, l_out, probs, labels, acc = self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                if self.training_step % 50 == 0:
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1

            except tf.errors.OutOfRangeError:

                m_iou = self.evaluate(dataset)
                if m_iou > np.max(self.mIou_list):
                    # Save the best model
                    snapshot_directory = join(self.saving_path, 'snapshots')
                    makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                self.mIou_list.append(m_iou)
                log_out('Best m_IoU is: {:5.3f}'.format(max(self.mIou_list)), self.Log_file)

                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)
                # Update learning rate
                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                self.sess.run(op)
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

        print('finished')
        self.sess.close()

    def evaluate(self, dataset):

        # Initialise iterator with validation data
        self.sess.run(dataset.val_init_op)

        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0

        for step_id in range(self.config.val_steps):
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = (self.prob_logits, self.labels, self.accuracy)
                stacked_prob, labels, acc = self.sess.run(ops, {self.is_training: False})
                pred = np.argmax(stacked_prob, 1)
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid - 1
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        return mean_iou

    def get_loss(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        output_loss = tf.reduce_mean(weighted_losses)
        return output_loss

    def bilateral_context_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
        """
        Inputs: 
        feature: [B, N, 1, c] input features
        xyz: [B, N, 3] input coordinates
        neigh_idx: [B, N, k] indices of k neighbors

        Output:
        output_feat: [B, N, 1, 2*d_out] encoded (output) features
        shifted_neigh_xyz: [B, N, k, 3] shifted neighbor coordinates, for augmentation loss
        """
        batch_size = tf.shape(xyz)[0]
        num_points = tf.shape(xyz)[1]

        # Input Encoding
        feature = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)

        # Bilateral Augmentation
        neigh_feat = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx) # B, N, k, d_out/2 
        neigh_xyz = self.gather_neighbour(xyz, neigh_idx) # B, N, k, 3
        tile_feat = tf.tile(feature, [1, 1, self.config.k_n, 1]) # B, N, k, d_out/2
        tile_xyz = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, self.config.k_n, 1]) # B, N, k, 3

        feat_info = tf.concat([neigh_feat - tile_feat, tile_feat], axis=-1) # B, N, k, d_out
        neigh_xyz_offsets = helper_tf_util.conv2d(feat_info, xyz.get_shape()[-1].value, [1, 1], name + 'mlp5', [1, 1], 'VALID', True, is_training) # B, N, k, 3       
        shifted_neigh_xyz = neigh_xyz + neigh_xyz_offsets # B, N, k, 3

        xyz_info = tf.concat([neigh_xyz - tile_xyz, shifted_neigh_xyz, tile_xyz], axis=-1) # B, N, k, 9
        neigh_feat_offsets = helper_tf_util.conv2d(xyz_info, feature.get_shape()[-1].value, [1, 1], name + 'mlp6', [1, 1], 'VALID', True, is_training) # B, N, k, d_out/2
        shifted_neigh_feat = neigh_feat + neigh_feat_offsets # B, N, k, d_out/2

        xyz_encoding = helper_tf_util.conv2d(xyz_info, d_out//2, [1, 1], name + 'mlp7', [1, 1], 'VALID', True, is_training) # B, N, k, d_out/2
        feat_info = tf.concat([shifted_neigh_feat, feat_info], axis=-1) # B, N, k, 3/2*d_out
        feat_encoding = helper_tf_util.conv2d(feat_info, d_out//2, [1, 1], name + 'mlp8', [1, 1], 'VALID', True, is_training) # B, N, k, d_out/2
        
        # Mixed Local Aggregation
        overall_info = tf.concat([xyz_encoding, feat_encoding], axis=-1) # B, N, k, d_out
        k_weights = helper_tf_util.conv2d(overall_info, overall_info.get_shape()[-1].value, [1, 1], name + 'mlp9', [1, 1], 'VALID', bn=False, activation_fn=None) # B, N, k, d_out
        k_weights = tf.nn.softmax(k_weights, axis=2) # B, N, k, d_out
        overall_info_weighted_sum = tf.reduce_sum(overall_info * k_weights, axis=2, keepdims=True) # B, N, 1, d_out
        overall_info_max = tf.reduce_max(overall_info, axis=2, keepdims=True) # B, N, 1, d_out
        overall_encoding = tf.concat([overall_info_max, overall_info_weighted_sum], axis=-1) # B, N, 1, 2*d_out

        # Output Encoding
        overall_encoding = helper_tf_util.conv2d(overall_encoding, d_out, [1, 1], name + 'mlp10', [1, 1], 'VALID', True, is_training) # B, N, 1, d_out
        output_feat = helper_tf_util.conv2d(overall_encoding, d_out * 2, [1, 1], name + 'mlp11', [1, 1], 'VALID',  True, is_training, activation_fn=tf.nn.leaky_relu) # B, N, 1, 2*d_out
        return output_feat, shifted_neigh_xyz

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features