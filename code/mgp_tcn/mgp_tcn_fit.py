#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Michael Moor October 2018. MGP-Module adapted from Futoma et al. (https://arxiv.org/abs/1706.04152)

"""
import faulthandler
import os.path 
import pickle
import sys
from time import time
import traceback

import numpy as np
from sacred import Experiment
from sklearn.metrics import roc_auc_score, average_precision_score
from tempfile import NamedTemporaryFile
import tensorflow as tf

from .memory_saving_gradients import gradients_memory
from ..preprocessing.main_preprocessing_mgp_tcn import load_data
from .tcn import CausalConv1D, TemporalBlock, TemporalConvNet
from .util import select_horizon, pad_rawdata_nomed, SE_kernel, OU_kernel, dot, \
    CG, Lanczos, block_CG, block_Lanczos
from .util import mask_large_samples as ev_mask

# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
tf.__dict__["gradients"] = gradients_memory
ex = Experiment('MGP-TCN')


def get_tcn_window(kernel_size, n_levels):
    window = 1
    for i in range(n_levels):
        window += 2**i * (kernel_size-1)
    return window


def mask_large_samples(data, thres, obs_min, static=None, return_mask=False):
    """Remove outliers by cutoff in order to fit data into memory (one outlier patient has 11k observation values)."""
    result_data = []
    n = len(data) #number of data views of compact format (values, times, indices, ..)
    mask = data[8] <= thres
    min_mask = data[8] >= obs_min #mask patients with less than n_mc_smps many num_obs_values
    print('-> {} patients have less than {} observation values'.format(np.sum(~min_mask),obs_min))
    mask = np.logical_and(mask, min_mask)
    print('---> Removing {} patients'.format(np.sum(~mask)))
    for i in np.arange(n):
        result_data.append(data[i][mask])
    if static is not None:
        result_static = static[mask]
        if return_mask:
            return result_data, result_static, mask 
        else:
            return result_data, result_static
    else:
        if return_mask:
            return result_data, mask
        else:
            return result_data


def count_parameters():
    """Count the number of trainable parameters."""
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        name = variable.name
        shape = variable.get_shape()
        #print(shape)
        #print(len(shape))
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        print(name, [dim for dim in shape], variable_parameters)
        total_parameters += variable_parameters
    print('Number of trainable parameters = {}'.format(total_parameters))


def compute_l1():
    #with tf.variable_scope("",reuse=True):
        #conv1_kernel_val = tf.get_variable('temporal_conv_net/tblock_0/conv1/kernel')
    kernel_name = 'temporal_conv_net/tblock_0/conv1/kernel:0'
    bias_name = 'temporal_conv_net/tblock_0/conv1/bias:0'
    gr = tf.get_default_graph()
    
    conv1_kernel_val = gr.get_tensor_by_name(kernel_name) #.eval(session=sess)
    conv1_bias_val = gr.get_tensor_by_name(bias_name)

    kernel_abs = tf.abs(conv1_kernel_val) # compute absolute of kernel tensor
    bias_abs = tf.abs(conv1_bias_val)

    kernel_abs_sum = tf.reduce_sum(kernel_abs, axis=0) # sum over kernel size, such that each entry represents absolute sum of all filter values for one channel and one filter
    bias_abs_sum = tf.reduce_sum(bias_abs)

    l1_per_filter = tf.reduce_sum(kernel_abs_sum, axis=0) #compute l1 over all channels per filter, since all elements are positive: sum over each channel
    l1_total = tf.reduce_sum(l1_per_filter) #sum over all filters of 1st convolution
    #add bias term to l1_total:
    l1_total += bias_abs_sum

    #normalize norm-value such that it is independent of the filter shape (s.t. a l1 penalty coefficient always has same impact)
    #compute total n of parameters of first conv1 layer:
    n_param=0
    for var in [conv1_kernel_val, conv1_bias_val]:
        shape = var.get_shape()
        n_var_param = 1.0
        for dim in shape:
            n_var_param *= dim.value
        n_param += n_var_param

    return l1_total/n_param


def compute_global_l2():
    variables = tf.trainable_variables()
    weights = [v for v in variables if 'kernel' in v.name ]
    L2 = tf.add_n([ tf.nn.l2_loss(w) for w in weights ])
    #print([w.name for w in weights])

    '''weight_names = [w.name for w in weights]
    values = sess.run(weight_names)
    weight_dims = [w.shape for w in values]'''
    
    weight_dims = [w.get_shape() for w in weights]

    n_weights = 0

    for weight_dim in weight_dims:
        n_weights_per_kernel = 1.0
        for dim in weight_dim:
            n_weights_per_kernel *= dim.value #dim 
        n_weights += n_weights_per_kernel

    print('N_weights:', n_weights)
    print('Weight Dims:', weight_dims)

    L2_per_weight = L2/n_weights
    #print(L2_per_weight.eval(session=sess))

    return L2_per_weight

#####
##### Convinience classes for managing parameters
#####

class DecompositionMethod():
    valid_methods = ['chol', 'cg']
    def __init__(self, methodname, add_diag=1e-3):
        if methodname not in self.valid_methods:
            raise ValueError('{} is not a valid methodname. Must be one of {}'.format(methodname, self.valid_methods))
        self.methodname = methodname
        self.add_diag = add_diag


class GPParameters():
    def __init__(self, input_dim, M, n_mc_smps):
        self.input_dim = input_dim
        self.M = M
        self.log_length = tf.Variable(tf.random_normal([1],mean=1,stddev=0.1),name="GP-log-length") 
        self.length = tf.exp(self.log_length)

        #different noise level of each lab
        self.log_noises = tf.Variable(tf.random_normal([input_dim],mean=-2,stddev=0.1),name="GP-log-noises")
        self.noises = tf.exp(self.log_noises)

        #init cov between labs
        self.L_f_init = tf.Variable(tf.eye(input_dim),name="GP-Lf")
        self.Lf = tf.matrix_band_part(self.L_f_init,-1,0)
        self.Kf = tf.matmul(self.Lf,tf.transpose(self.Lf))

        self.n_mc_smps = n_mc_smps


# Model-specific prepro function to reset times to 0 to 48 (counted from first provided measurement):
@ex.capture
def reset_times(train_data,validation_data,test_data):
    train, val, test = train_data.copy(), validation_data.copy(), test_data.copy()
    for dataset in [train, val, test]:
        
        times = dataset[1].copy()
        num_rnn_grid_times = []
        rnn_grid_times = []
        for i in range(len(times)):
            times[i] = times[i]-min(times[i])
            end_time = times[i][-1]
            num_rnn_grid_time = int(np.floor(end_time)+1)
            num_rnn_grid_times.append(num_rnn_grid_time)
            rnn_grid_times.append(np.arange(num_rnn_grid_time))
        dataset[1] = times
        dataset[5] = np.array(num_rnn_grid_times)
        dataset[6] = np.array(rnn_grid_times)

    return train, val, test    

#####
##### Tensorflow functions 
#####
@ex.capture
def draw_GP(Yi,Ti,Xi,ind_kfi,ind_kti,method, gp_params):
    """ 
    given GP hyperparams and data values at observation times, draw from 
    conditional GP
    
    inputs:
        length,noises,Lf,Kf: GP params
        Yi: observation values
        Ti: observation times
        Xi: grid points (new times for rnn)
        ind_kfi,ind_kti: indices into Y
    returns:
        draws from the GP at the evenly spaced grid times Xi, given hyperparams and data
    """  
    n_mc_smps, length, noises, Lf, Kf = gp_params.n_mc_smps, gp_params.length, gp_params.noises, gp_params.Lf, gp_params.Kf
    M = gp_params.M
    ny = tf.shape(Yi)[0]
    K_tt = OU_kernel(length,Ti,Ti)
    D = tf.diag(noises)

    grid_f = tf.meshgrid(ind_kfi,ind_kfi) #same as np.meshgrid
    Kf_big = tf.gather_nd(Kf,tf.stack((grid_f[0],grid_f[1]),-1))
    
    grid_t = tf.meshgrid(ind_kti,ind_kti) 
    Kt_big = tf.gather_nd(K_tt,tf.stack((grid_t[0],grid_t[1]),-1))

    Kf_Ktt = tf.multiply(Kf_big,Kt_big)

    DI_big = tf.gather_nd(D,tf.stack((grid_f[0],grid_f[1]),-1))
    DI = tf.diag(tf.diag_part(DI_big)) #D kron I
    
    #data covariance. 
    #Either need to take Cholesky of this or use CG / block CG for matrix-vector products
    Ky = Kf_Ktt + DI + method.add_diag*tf.eye(ny)   

    ### build out cross-covariances and covariance at grid
    
    nx = tf.shape(Xi)[0]
    
    K_xx = OU_kernel(length,Xi,Xi)
    K_xt = OU_kernel(length,Xi,Ti)
                       
    ind = tf.concat([tf.tile([i],[nx]) for i in range(M)],0)
    grid = tf.meshgrid(ind,ind)
    Kf_big = tf.gather_nd(Kf,tf.stack((grid[0],grid[1]),-1))
    ind2 = tf.tile(tf.range(nx),[M])
    grid2 = tf.meshgrid(ind2,ind2)
    Kxx_big =  tf.gather_nd(K_xx,tf.stack((grid2[0],grid2[1]),-1))
    
    K_ff = tf.multiply(Kf_big,Kxx_big) #cov at grid points           
                 
    full_f = tf.concat([tf.tile([i],[nx]) for i in range(M)],0)            
    grid_1 = tf.meshgrid(full_f,ind_kfi,indexing='ij')
    Kf_big = tf.gather_nd(Kf,tf.stack((grid_1[0],grid_1[1]),-1))
    full_x = tf.tile(tf.range(nx),[M])
    grid_2 = tf.meshgrid(full_x,ind_kti,indexing='ij')
    Kxt_big = tf.gather_nd(K_xt,tf.stack((grid_2[0],grid_2[1]),-1))

    K_fy = tf.multiply(Kf_big,Kxt_big)
       
    #now get draws!
    y_ = tf.reshape(Yi,[-1,1])
    
    xi = tf.random_normal((nx*M, n_mc_smps))
    #print('xi shape:')
    #print(xi.shape)
    
    if method.methodname == 'chol':
        Ly = tf.cholesky(Ky)
        Mu = tf.matmul(K_fy,tf.cholesky_solve(Ly,y_))
        Sigma = K_ff - tf.matmul(K_fy,tf.cholesky_solve(Ly,tf.transpose(K_fy))) + method.add_diag*tf.eye(tf.shape(K_ff)[0]) 
        draw = Mu + tf.matmul(tf.cholesky(Sigma),xi) 
        draw_reshape = tf.transpose(tf.reshape(tf.transpose(draw),[n_mc_smps,M,nx]),perm=[0,2,1])

    elif method.methodname == 'cg':
        Mu = tf.matmul(K_fy,CG(Ky,y_)) #May be faster with CG for large problems
        #Never need to explicitly compute Sigma! Just need matrix products with Sigma in Lanczos algorithm
        def Sigma_mul(vec):
            # vec must be a 2d tensor, shape (?,?) 
            return tf.matmul(K_ff,vec) - tf.matmul(K_fy,block_CG(Ky,tf.matmul(tf.transpose(K_fy),vec))) 
        def large_draw():             
            return Mu + block_Lanczos(Sigma_mul,xi,n_mc_smps) #no need to explicitly reshape Mu
        #draw = tf.cond(tf.less(nx*M,BLOCK_LANC_THRESH),small_draw,large_draw)
        draw = large_draw()
        draw_reshape = tf.transpose(tf.reshape(tf.transpose(draw),[n_mc_smps,M,nx]),perm=[0,2,1])
        #print('cg draw shape:')
        #print(draw_reshape.shape)   

    #TODO: it's worth testing to see at what point computation speedup of Lanczos algorithm is useful & needed.
    # For smaller examples, using Cholesky will probably be faster than this unoptimized Lanczos implementation.
    # Likewise for CG and BCG vs just taking the Cholesky of Ky once
    
    #draw_reshape = tf.transpose(tf.reshape(tf.transpose(draw),[n_mc_smps,M,nx]),perm=[0,2,1])
    return draw_reshape    

@ex.capture        
def get_GP_samples(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
                   num_rnn_grid_times, cov_grid, input_dim,method, gp_params, lab_vitals_only, pad_before): ##,med_cov_grid
    """
    returns samples from GP at evenly-spaced gridpoints
    """ 

    n_mc_smps, M = gp_params.n_mc_smps, gp_params.M
    grid_max = tf.shape(X)[1]
    Z = tf.zeros([0,grid_max,input_dim])
    
    N = tf.shape(T)[0] #number of observations
        
    #setup tf while loop (have to use this bc loop size is variable)
    def cond(i,Z):
        return i<N
    
    def body(i,Z):
        Yi = tf.reshape(tf.slice(Y,[i,0],[1,num_obs_values[i]]),[-1]) #MM: tf.reshape(x, [-1]) flattens tensor x (e.g. [2,3,1] to [6]), slice cuts out all Y data of one patient
        Ti = tf.reshape(tf.slice(T,[i,0],[1,num_obs_times[i]]),[-1])
        ind_kfi = tf.reshape(tf.slice(ind_kf,[i,0],[1,num_obs_values[i]]),[-1])
        ind_kti = tf.reshape(tf.slice(ind_kt,[i,0],[1,num_obs_values[i]]),[-1])
        Xi = tf.reshape(tf.slice(X,[i,0],[1,num_rnn_grid_times[i]]),[-1])
        X_len = num_rnn_grid_times[i]
                
        GP_draws = draw_GP(Yi,Ti,Xi,ind_kfi,ind_kti,method=method, gp_params=gp_params)
        pad_len = grid_max-X_len #pad by this much
        #padding direction:
        if pad_before:
            print('Padding GP_draws before observed data..')
            padded_GP_draws = tf.concat([tf.zeros((n_mc_smps,pad_len,M)), GP_draws],1) 
        else:
            padded_GP_draws = tf.concat([GP_draws,tf.zeros((n_mc_smps,pad_len,M))],1) 

        if lab_vitals_only:
            Z = tf.concat([Z,padded_GP_draws],0) #without covs
        else: #with covs
            medcovs = tf.slice(cov_grid,[i,0,0],[1,-1,-1])
            tiled_medcovs = tf.tile(medcovs,[n_mc_smps,1,1])
            padded_GPdraws_medcovs = tf.concat([padded_GP_draws,tiled_medcovs],2)
            Z = tf.concat([Z,padded_GPdraws_medcovs],0) #with covs
        
        return i+1,Z  
    
    i = tf.constant(0)
    #with tf.control_dependencies([tf.Print(tf.shape(ind_kf), [tf.shape(ind_kf), tf.shape(ind_kt), num_obs_values], 'ind_kf & ind_kt & num_obs_values')]):
    i,Z = tf.while_loop(cond,body,loop_vars=[i,Z],
            shape_invariants=[i.get_shape(),tf.TensorShape([None,None,None])])

    return Z

# TODO: MODIFY THIS FUNCTION FOR TCNs
@ex.capture
def get_preds(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
              num_rnn_grid_times, cov_grid, input_dim,method, gp_params, tcn, is_training, n_classes, lab_vitals_only, pad_before, losstype): #med_cov_grid
    """
    helper function. takes in (padded) raw datas, samples MGP for each observation, 
    then feeds it all through the TCN to get predictions.
    
    inputs:
        Y: array of observation values (labs/vitals). batchsize x batch_maxlen_y
        T: array of observation times (times during encounter). batchsize x batch_maxlen_t
        ind_kf: indiceste into each row of Y, pointing towards which lab/vital. same size as Y
        ind_kt: indices into each row of Y, pointing towards which time. same size as Y
        num_obs_times: number of times observed for each encounter; how long each row of T really is
        num_obs_values: number of lab values observed per encounter; how long each row of Y really is
        num_rnn_grid_times: length of even spaced RNN grid per encounter
                    
    returns:
        predictions (unnormalized log probabilities) for each MC sample of each obs
    """
    
    Z = get_GP_samples(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
                       num_rnn_grid_times, cov_grid, input_dim, method=method, gp_params=gp_params, lab_vitals_only=lab_vitals_only, pad_before=pad_before) #batchsize*num_MC x batch_maxseqlen x num_inputs    ##,med_cov_grid
    Z.set_shape([None,None,input_dim]) #somehow lost shape info, but need this
    N = tf.shape(T)[0] #number of observations 
    
    tcn_logits = tf.layers.dense(
        tcn(Z, training=is_training)[:, -1, :],
        n_classes, activation=None, 
        kernel_initializer=tf.orthogonal_initializer(),
        name='last_linear', reuse=(losstype == 'average') 
    ) # reuse should be true if losstype is average

    return tcn_logits

@ex.capture
def get_losses(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
              num_rnn_grid_times, cov_grid, input_dim,method, gp_params, tcn, is_training, n_classes, lab_vitals_only, pad_before,
              labels, pos_weight): #med_cov_grid
    """
    helper function. takes in (padded) raw datas, samples MGP for each observation, 
    then feeds it all through the TCN to get predictions.
    
    inputs:
        Y: array of observation values (labs/vitals). batchsize x batch_maxlen_y
        T: array of observation times (times during encounter). batchsize x batch_maxlen_t
        ind_kf: indiceste into each row of Y, pointing towards which lab/vital. same size as Y
        ind_kt: indices into each row of Y, pointing towards which time. same size as Y
        num_obs_times: number of times observed for each encounter; how long each row of T really is
        num_obs_values: number of lab values observed per encounter; how long each row of Y really is
        num_rnn_grid_times: length of even spaced RNN grid per encounter
                    
    returns:
        predictions (unnormalized log probabilities) for each MC sample of each obs
    """
    
    Z = get_GP_samples(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
                       num_rnn_grid_times, cov_grid, input_dim, method=method, gp_params=gp_params, lab_vitals_only=lab_vitals_only, pad_before=pad_before) #batchsize*num_MC x batch_maxseqlen x num_inputs    ##,med_cov_grid
    Z.set_shape([None,None,input_dim]) #somehow lost shape info, but need this
    N = tf.shape(Z)[0] #number of observations 


    # We only want to consider up tw0 7 timepoints before end
    T_max = 7
    
    # Only during training we want to average over the last few predictions in order to give
    # the model the incentive to predict early
    tcn_out = tcn(Z, training=is_training)[:, -T_max:, :]
    tcn_logits = tf.layers.dense(tcn_out,
        n_classes, activation=None, 
        kernel_initializer=tf.orthogonal_initializer(),
        name='last_linear', reuse=False
    )

    # Only get a few of the last obs
    #used_grid = tf.reduce_min(tf.stack([num_rnn_grid_times, tf.fill(tf.shape(num_rnn_grid_times), T_max)]), axis=0)
    #tiled = tf.tile(tf.expand_dims(used_grid, axis=-1), [1, gp_params.n_mc_smps])
    #expanded_used_grid = tf.reshape(tiled, [-1])
    tiled_labels = tf.tile(tf.expand_dims(labels, axis=1), tf.stack([1, T_max, 1]))
    all_losses = tf.nn.weighted_cross_entropy_with_logits(logits=tcn_logits,targets=tiled_labels, pos_weight=pos_weight)
    average_losses = tf.reduce_mean(all_losses, axis=-1)

    return average_losses


@ex.capture
def get_probs_and_accuracy(preds, O, n_mc_smps):
    """
    helper function. we have a prediction for each MC sample of each observation
    in this batch.  need to distill the multiple preds from each MC into a single
    pred for this observation.  also get accuracy. use true probs to get ROC, PR curves in sklearn
    """
    all_probs = tf.exp(preds[:,1] - tf.reduce_logsumexp(preds, axis = 1)) #normalize; and drop a dim so only prob of positive case
    N = tf.cast(tf.shape(preds)[0]/n_mc_smps,tf.int32) #actual number of observations in preds, collapsing MC samples                    
    
    #predicted probability per observation; collapse the MC samples
    probs = tf.zeros([0]) #store all samples in a list, then concat into tensor at end
    #setup tf while loop (have to use this bc loop size is variable)
    def cond(i,probs):
        return i < N
    def body(i,probs):
        probs = tf.concat([probs,[tf.reduce_mean(tf.slice(all_probs,[i*n_mc_smps],[n_mc_smps]))]],0)
        return i+1,probs    
    i = tf.constant(0)
    i,probs = tf.while_loop(cond,body,loop_vars=[i,probs],shape_invariants=[i.get_shape(),tf.TensorShape([None])])
        
    #compare to truth; just use cutoff of 0.5 for right now to get accuracy
    correct_pred = tf.equal(tf.cast(tf.greater(probs,0.5),tf.int32), O)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 
    return probs,accuracy


@ex.config
def mpg_rnn_config():
    dataset = {
        'na_thres': 500, #30,100 numb of non-nans a variable must show in prepro for not being dropped (for dropping too rarely sampled variables)
        'datapath': 'output/',
        'overwrite': 0, #0 #indicates whether preproscript should run through, otherwise load previous dump.
        'horizon': 0,
        'data_sources': ['labs','vitals','covs'], #future: meds #labs and vitals required currently! #TODO: make prepro script more modular s.t. labs and vitals can be left away
        'lab_vitals_only': True, # Flag for only using lab vitals and no covs in get_preds, get_GP_samples..
        'min_length': 7, #None #Minimal num of gridpoints below which sample is dropped
        'max_length': 200, #None #Maximal num of gridpoints above which sample is dropped (5 outliers lead to 3-4x memory use)
        'min_pad_length': 8, #None Minimal time series length in padded batches which is max(max_batch_len, min_pad_length) such that no batch is shorter than this parameter (such that TCN works!)
        'num_obs_thres': 10000, #None, Min and Max number of values (patients with 0 values appear due to control matching, drop controls with no values..) currently, 10k drops 1 patient, this is for memory saving.
        'split':0 # 0-4
    }
    #Padding method:
    pad_before=False #zero padding of batches before observed data, instead of after if False.

    #GP method:
    decomposition_method='chol' #'cg' 
    add_diag = 1e-1 #changed from 1e-3 for more stability
    losstype='weighted' # 'regular'
    
    batch_size = 100 #NOTE may want to play around with this
    learning_rate = 0.001
    decay_learning_rate = False
    
    training_iters = 30 #num epochs

    levels=4
    kernel_size=4
    n_hidden = 40 # hidden layer num of features; assumed same
    #n_layers = 1 #3 number of layers of stacked LSTMs
    n_mc_smps = 10 #10
    dropout = 0.1 #not keep-dropout, but % of dropping out!
    reduction_dim = None  # Reduction dim for 1x1 conv in temporal blocks. Set to None to disable
    drop_first_res = False #When applying l1 / sparsity, drop first residual connection in TemporalConvnet s.t. all responsibility with 1st convolution filters
    l1_filter_reg = None # regularize first conv layer for sparse and interpretable filters..
    L1_penalty = 0 # coefficient for l1 norm of first filter in loss
    L2_penalty = None # using per-weight norm! hence lambda so large.. multiplied with start per weight-norm value arrives at around 10. train loss around 100-2000
    #Configuration: reseting time to 0-48 instead of using raw hour from in-time (e.g. 210.5 - 258.5)
    time_reset = 0 #1

@ex.named_config
def decay_lr():
    learning_rate = 0.01
    decay_learning_rate = True


@ex.capture(prefix='dataset')
def get_dataset(na_thres, datapath, overwrite, horizon, data_sources, min_length, max_length, split, num_obs_thres):
    print('USING SPLIT {}'.format(split))    
    datapath += 'mgp-rnn-datadump_'+'_'.join([str(el) for el in data_sources])+'_na_thres_{}_min_length_{}_max_length_{}_horizon_0_split_{}.pkl'.format(na_thres, min_length, max_length, split)
    if (overwrite or not os.path.isfile(datapath) ): #check if data was not prepared and loaded before:
        if overwrite:
            print('Overwriting mode: running prepro script, dumping and loading data..')
        else:
            print('Data dump not found. Running prepro script, dumping and loading data..')
        #Preprocess and load data..
        full_dataset = load_data(test_size=0.1, horizon=0, na_thres=na_thres, variable_start_index = 5, data_sources=data_sources, min_length=min_length, max_length=max_length, split=split)
        #Then dump it, for faster iterating
        pickle.dump( full_dataset, open(datapath, "wb"))
    else:
        print('Loading existing preprocessed data dump')
        full_dataset = pickle.load( open( datapath, "rb" ))

    if horizon !=0:
        train_data,validation_data,test_data = full_dataset[1:4]
        train_static_data, validation_static_data, test_static_data = full_dataset[4:7]
        #first masking (necessary for select_horizon function to work)
        obs_min = 10 #hard-coded for now, but we never used different value due to constant mc_smps..

        print('First masking before horizon selection')
        stats_train, train_data, train_static_data = ev_mask(train_data, num_obs_thres, obs_min, static=train_static_data)
        stats_validation, validation_data, validation_static_data = ev_mask(validation_data, num_obs_thres, obs_min, static=validation_static_data)
        stats_test, test_data, test_static_data = ev_mask(test_data, num_obs_thres, obs_min, static=test_static_data)
        
        train_data = select_horizon(train_data, horizon)
        validation_data = select_horizon(validation_data, horizon)
        test_data = select_horizon(test_data, horizon)
        full_dataset = full_dataset[:1] + \
            [train_data, validation_data, test_data, train_static_data, validation_static_data, test_static_data]

    return full_dataset

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def print_var_statistics(name, values):
    print(f'{name}: mean: {np.mean(values)} +/- {np.std(values)}, median: {np.median(values)} +/- {mad(values)}')

@ex.command(prefix='dataset')
def dataset_statistics(na_thres, datapath, horizon, data_sources, min_length, max_length, split):
    print('USING SPLIT {}'.format(split))    
    datapath += 'mgp-rnn-datadump_'+'_'.join([str(el) for el in data_sources])+'_na_thres_{}_min_length_{}_max_length_{}_horizon_{}_split_{}.pkl'.format(na_thres, min_length, max_length, horizon, split)
    
    print('Loading existing preprocessed data dump')
    #train_data, validation_data, test_data, variables = pickle.load( open( datapath, "rb" ))
    full_dataset = pickle.load( open( datapath, "rb" ))
    train_data, validation_data, test_data = full_dataset[1:4]
    for name, data in zip(['train', 'val', 'test'], [train_data, validation_data, test_data]):
        values = data[0]
        times = data[1]
        ind_lvs = data[2]
        ind_times = data[3]
        labels = data[4]
        num_rnn_grid_times = data[5]
        rnn_grid_times = data[6]
        num_obs_times = data[7]
        num_obs_values = data[8]

        print(f'{name}')
        print_var_statistics('num_obs_times', num_obs_times)
        print_var_statistics('num_obs_values', num_obs_values)
        print_var_statistics('num_rnn_grid_times', num_rnn_grid_times)


@ex.main
def fit_mgp_tcn(decomposition_method, add_diag, losstype, n_hidden, levels, kernel_size, n_mc_smps, dropout, reduction_dim, batch_size, learning_rate, decay_learning_rate,
                training_iters, time_reset, l1_filter_reg, drop_first_res, L1_penalty, L2_penalty, pad_before, _rnd, _seed, _run, dataset):
    #Parameters (hard-coded for prototyping)
    if len(_run.observers) > 0:
         checkpoint_path = os.path.join(_run.observers[0].dir, 'model_checkpoints')
    else:
        checkpoint_path = 'model_checkpoints'

    rs = _rnd #fixed seed in np

    tf.logging.set_verbosity(tf.logging.ERROR)

    # Load dataset
    full_dataset = get_dataset() # load dataset
    data_sources = dataset['data_sources'] #get names of data sources (labs, vitals ,..)
    lab_vitals_only = dataset['lab_vitals_only']
    num_obs_thres = dataset['num_obs_thres']
    min_pad_length = dataset['min_pad_length'] #minimal time series length a batch should be padded to (for TCNs..)
    obs_min = np.max([10, n_mc_smps]) #remove the samples with less than 10 observation values, this lanczos impl fails when mc_smps > num_obs
    method = DecompositionMethod(decomposition_method, add_diag)

    print('Data is loaded. Now assign available data sources to variables')
    if lab_vitals_only:
        print('Only lab and vitals will be used..')
    #TODO: assign items of datasetlist to variable names as line below! each data_source has 3 splits (except labvitals in one currently)
    index = 0
    variables = full_dataset[index] #variable list always first element
    index += 1 # go to next element in full_dataset
    if 'labs' and 'vitals' in data_sources:
        train_data,validation_data,test_data = full_dataset[index:index+3]
        index+=3
    else:
        raise ValueError('Labs or Vitals not selected in config, yet they are required') #TODO: make this case possible
    if 'covs' in data_sources:
        train_static_data, validation_static_data, test_static_data = full_dataset[index:index+3]
        if num_obs_thres is not None:
            train_data, train_static_data = mask_large_samples(train_data, num_obs_thres, obs_min, static=train_static_data)
            validation_data, validation_static_data = mask_large_samples(validation_data, num_obs_thres, obs_min, static=validation_static_data)
            test_data, test_static_data = mask_large_samples(test_data, num_obs_thres, obs_min, static=test_static_data)
    elif 'labs' or 'vitals' in data_soures:
        if num_obs_thres is not None:
            train_data = mask_large_samples(train_data, num_obs_thres, obs_min)
            validation_data = mask_large_samples(validation_data, num_obs_thres, obs_min)
            test_data = mask_large_samples(test_data, num_obs_thres, obs_min)

    # Check if we reset_times:
    if time_reset:
        # for all splits, reset times to 0 - 48 hours, rnn_grid points accordingly!
        train_data,validation_data,test_data = reset_times(train_data,validation_data,test_data)
        print('Time was reset to [0, 48] hours')

    M = len(variables)  
    Ntr = len(train_data[0])
    Nva = len(validation_data[0])
    n_covs = train_static_data.shape[1]
    print('n_covs = {}'.format(n_covs))
    print('data sources: {}'.format(data_sources))
    print('covs in data_sources: {}'.format('covs' in data_sources))

    #Assign data splits:
    values_tr = train_data[0]; values_va = validation_data[0]
    times_tr = train_data[1]; times_va = validation_data[1]
    ind_lvs_tr = train_data[2]; ind_lvs_va = validation_data[2]
    ind_times_tr = train_data[3]; ind_times_va = validation_data[3]
    labels_tr = train_data[4]; labels_va = validation_data[4]
    num_rnn_grid_times_tr = train_data[5]; num_rnn_grid_times_va = validation_data[5]
    rnn_grid_times_tr = train_data[6]; rnn_grid_times_va = validation_data[6]
    num_obs_times_tr = train_data[7]; num_obs_times_va = validation_data[7]
    num_obs_values_tr = train_data[8]; num_obs_values_va = validation_data[8]
    
    if 'covs' in data_sources:
        covs_tr = train_static_data 
        covs_va = validation_static_data #; covs_te = test_static_data    

    #Get class imbalance (for weighted loss):
    case_prev = labels_tr.sum()/float(len(labels_tr)) #get prevalence of cases in train dataset
    class_imb = 1/case_prev #class imbalance to use as class weight if losstype='weighted'


    print("data fully setup!")    
    sys.stdout.flush()

    #print('EXITING AFTER LOADING DATA')
    #sys.exit()
    
    #####
    ##### Setup model and graph
    ##### 
    
    # Learning Parameters
    
    decay_step = int(Ntr/batch_size) #100 #after how many batches will the learning rate be adjusted..
    
    ##test_freq     = Ntr/batch_size #eval on test set after this many batches
    test_freq = int(Ntr/batch_size / 4)
    # Network Parameters
    n_classes = 2 #binary outcome
    
    if lab_vitals_only:
        input_dim = M
    elif 'covs' in data_sources:
        input_dim = M + n_covs #dimensionality of input sequence. ## M+n_meds+n_covs
    else:
        input_dim = M 
    
    # Create graph
    # If we were to reset the default graph here, then we cannot control randomness

    #Experiment for trying to reproduce randomness..
    tf.set_random_seed(_seed)

    sess = tf.Session()

    #define decaying learning rate:
    global_step = tf.Variable(0, trainable=False) #Cave, had to add it to Adam loss min()!
    if decay_learning_rate:
        learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                                   decay_step , 0.96, staircase=True)

    ##### tf Graph - inputs 
    
    #observed values, times, inducing times; padded to longest in the batch
    Y = tf.placeholder("float", [None,None]) #batchsize x batch_maxdata_length
    T = tf.placeholder("float", [None,None]) #batchsize x batch_maxdata_length
    ind_kf = tf.placeholder(tf.int32, [None,None]) #index tasks in Y vector
    ind_kt = tf.placeholder(tf.int32, [None,None]) #index inputs in Y vector
    X = tf.placeholder("float", [None,None]) #grid points. batchsize x batch_maxgridlen
    cov_grid = tf.placeholder("float", [None,None,n_covs]) #combine w GP smps to feed into RNN ## n_meds+n_covs
    
    O = tf.placeholder(tf.int32, [None]) #labels. input is NOT as one-hot encoding; convert at each iter
    num_obs_times = tf.placeholder(tf.int32, [None]) #number of observation times per encounter 
    num_obs_values = tf.placeholder(tf.int32, [None]) #number of observation values per encounter 
    num_rnn_grid_times = tf.placeholder(tf.int32, [None]) #length of each grid to be fed into RNN in batch
    
    N = tf.shape(Y)[0]                         
                                                                                                                                                                                      
    #also make O one-hot encoding, for the loss function
    O_dupe_onehot = tf.one_hot(tf.reshape(tf.tile(tf.expand_dims(O,1),[1,n_mc_smps]),[N*n_mc_smps]),n_classes)

    gp_params = GPParameters(input_dim, M, n_mc_smps) #changed M to input_dim, for only defining/setting input_dim once!

    #Define TCN Network:
    
    calculated_length = get_tcn_window(kernel_size, levels)
    if calculated_length > min_pad_length:
        print('Timeseries min_pad_length: {} are too short for Specified TCN Parameters requiring {}'.format(min_pad_length, calculated_length))
        min_pad_length = calculated_length
        print('>>>>>> Setting min_pad_length to: {}'.format(min_pad_length))

    #initialize architecture:
    tcn = TemporalConvNet([n_hidden] * levels, kernel_size, dropout, reduction_dim=reduction_dim, drop_first_res=drop_first_res)
    is_training = tf.placeholder("bool")
    
    #define tcn outputs:
    ##### Get predictions and feed into optimization
    if losstype=='average':
        losses = get_losses(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,num_rnn_grid_times, cov_grid, input_dim,
                      method=method, gp_params=gp_params, tcn=tcn, is_training=is_training, n_classes=n_classes, lab_vitals_only=lab_vitals_only, pad_before=pad_before,
                      labels=O_dupe_onehot, pos_weight=class_imb)
        loss_fit = tf.reduce_sum(losses)
    
    preds = get_preds(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,num_rnn_grid_times, cov_grid, input_dim,
                      method=method, gp_params=gp_params, tcn=tcn, is_training=is_training, n_classes=n_classes, lab_vitals_only=lab_vitals_only, pad_before=pad_before, losstype=losstype)    #med_cov_grid
    probs,accuracy = get_probs_and_accuracy(preds,O)
    
    # Define optimization problem
    if losstype=='weighted':
        loss_fit = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(logits=preds,targets=O_dupe_onehot, pos_weight=class_imb))
    if L2_penalty is not None:
        loss_reg = compute_global_l2() # normalized per weight! hence, use large lambda!
        loss = loss_fit + loss_reg*L2_penalty
    if l1_filter_reg is not None:
        loss_reg = compute_l1()
        if L2_penalty is not None:
            loss = loss + L1_penalty*loss_reg
        else:
            loss = loss_fit + L1_penalty*loss_reg
    if (L2_penalty is None) and (l1_filter_reg is None):
        loss = loss_fit

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step) ## added global step to minimize()
   
    ##### Initialize globals and get ready to start!
    sess.run(tf.global_variables_initializer())
    print("Graph setup!")

    #for more details, uncomment:
    #count_parameters()
    
    ### Add runoptions for memory issues:
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)

    #setup minibatch indices for training:
    starts = np.arange(0,Ntr,batch_size)
    ends = np.arange(batch_size,Ntr+1,batch_size)
    if ends[-1]<Ntr: 
        ends = np.append(ends,Ntr)
    num_batches = len(ends)

    #setup minibatch indices for validation (memory saving)
    va_starts = np.arange(0,Nva,batch_size)
    va_ends = np.arange(batch_size,Nva+1,batch_size)
    if va_ends[-1]<Nva: 
        va_ends = np.append(va_ends,Nva)

    
    #here: initial position of validation padding..
    T_pad_va, Y_pad_va, ind_kf_pad_va, ind_kt_pad_va, X_pad_va, cov_pad_va = pad_rawdata_nomed(
        times_va, values_va, ind_lvs_va, ind_times_va,
        rnn_grid_times_va, covs_va, num_rnn_grid_times_va, min_pad_length)

    ##### Main training loop
    saver = tf.train.Saver(max_to_keep = None)

    total_batches = 0
    best_val = 0
    for i in range(training_iters):
        #print max memory usage up to now
        print('Max Memory Usage up to now')
        print(sess.run(tf.contrib.memory_stats.MaxBytesInUse()))
        
        #train
        epoch_start = time()
        print("Starting epoch "+"{:d}".format(i))
        perm = rs.permutation(Ntr)
        batch = 0 
        for s,e in zip(starts,ends):
            if decay_learning_rate:
                print('Currrent Learning Rate:')
                print(sess.run(learning_rate))

            batch_start = time()
            inds = perm[s:e]
            T_pad,Y_pad,ind_kf_pad,ind_kt_pad,X_pad, cov_pad = pad_rawdata_nomed(
                    times_tr[inds],values_tr[inds],ind_lvs_tr[inds],ind_times_tr[inds],
                    rnn_grid_times_tr[inds], covs_tr[inds,:], num_rnn_grid_times_tr[inds], min_pad_length) ## meds_on_grid_tr[inds],covs_tr[inds,:]

            feed_dict={Y:Y_pad,T:T_pad,ind_kf:ind_kf_pad,ind_kt:ind_kt_pad,X:X_pad, cov_grid:cov_pad,
               num_obs_times:num_obs_times_tr[inds],
               num_obs_values:num_obs_values_tr[inds],
               num_rnn_grid_times:num_rnn_grid_times_tr[inds],O:labels_tr[inds], is_training: True} ##med_cov_grid:meds_cov_pad,                        

            try:        
                loss_,_ = sess.run([loss,train_op],feed_dict, options=run_options)
                
            except Exception as e:
                traceback.format_exc()
                print('Error occured in tensorflow during training:', e)
                #In addition dump more detailed traceback to txt file:
                with NamedTemporaryFile(suffix='.csv') as f:
                    faulthandler.dump_traceback(f)
                    _run.add_artifact(f.name, 'faulthandler_dump.csv')
                break
            
            print("Batch "+"{:d}".format(batch)+"/"+"{:d}".format(num_batches)+\
                  ", took: "+"{:.3f}".format(time()-batch_start)+", loss: "+"{:.5f}".format(loss_))
            sys.stdout.flush()
            batch += 1; total_batches += 1

            if total_batches % test_freq == 0: #Check val set every so often for early stopping
                print('--> Entering validation step...')
                #TODO: may also want to check validation performance at additional X hours back
                #from the event time, as well as just checking performance at terminal time
                #on the val set, so you know if it generalizes well further back in time as well 
                val_t = time()
                #Batch-wise Validation Phase:
                va_probs_tot = np.array([])
                va_perm = rs.permutation(Nva)
                va_labels_tot = labels_va[va_perm]
                for v_s,v_e in zip(va_starts,va_ends):
                    va_inds = va_perm[v_s:v_e]
                    va_feed_dict={Y:Y_pad_va[va_inds,:], T:T_pad_va[va_inds,:], ind_kf:ind_kf_pad_va[va_inds,:], 
                                ind_kt:ind_kt_pad_va[va_inds,:], X:X_pad_va[va_inds,:],
                                cov_grid:cov_pad_va[va_inds,:,:], num_obs_times:num_obs_times_va[va_inds],
                                num_obs_values:num_obs_values_va[va_inds], num_rnn_grid_times:num_rnn_grid_times_va[va_inds],
                                O:labels_va[va_inds], is_training: False}
                                                          
                    try:        
                        va_probs,va_acc,va_loss = sess.run([probs,accuracy,loss],va_feed_dict, options=run_options)     
                    except Exception as e:
                        traceback.format_exc()
                        print('Error occured in tensorflow during evaluation:', e)
                        break
                    #append current validation auprc to array of entire validation set
                    va_probs_tot = np.concatenate([va_probs_tot, va_probs])

                va_auc = roc_auc_score(va_labels_tot, va_probs_tot)
                va_prc = average_precision_score(va_labels_tot, va_probs_tot)   
                best_val = max(va_prc, best_val)
                print("Epoch "+str(i)+", seen "+str(total_batches)+" total batches. Validating Took "+\
                      "{:.2f}".format(time()-val_t)+\
                      ". OOS, "+str(0)+" hours back: Loss: "+"{:.5f}".format(va_loss)+ \
                      ", AUC: {:.5f}".format(va_auc)+", AUPR: "+"{:.5f}".format(va_prc))
                _run.log_scalar('train_loss', loss_, total_batches)
                _run.log_scalar('val_auprc', va_prc, total_batches)
                sys.stdout.flush()    
            
                #create a folder and put model checkpoints there
                saver.save(sess, checkpoint_path + "/epoch_{}".format(i), global_step=total_batches)
        print("Finishing epoch "+"{:d}".format(i)+", took "+\
              "{:.3f}".format(time()-epoch_start))     
        
    return {'Best Validation AUPRC': best_val}

if __name__ == '__main__':
    ex.run_commandline()
