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
from .util import pad_rawdata_nomed, SE_kernel, OU_kernel, dot, CG, Lanczos, block_CG, block_Lanczos 

# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
tf.__dict__["gradients"] = gradients_memory

ex = Experiment('MGP-TCN')

def get_tcn_window(kernel_size, n_levels):
    window = 1
    for i in range(n_levels):
        window += 2**i * (kernel_size-1)
    return window

#utility function to count the number of trainable parameters:
def count_parameters():
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

#Sim dataset functions from mgp_tcn_fit.py (jfutoma)
def gen_MGP_params(M, rs):
    """
    Generate some MGP params for each class. 
    Assume MGP is stationary and zero mean, so hyperparams are just:
        Kf: covariance across time series
        length: length scale for shared kernel across time series
        noise: noise level for each time series
    """ 
    
    true_Kfs = []
    true_noises = []
    true_lengths = []
    
    #Class 0
    tmp = rs.normal(0,.2,(M,M))
    true_Kfs.append(np.dot(tmp,tmp.T))
    true_lengths.append(1.0)
    true_noises.append(np.linspace(.02,.08,M))
    
    #Class 1
    tmp = rs.normal(0,.4,(M,M))
    true_Kfs.append(np.dot(tmp,tmp.T))
    true_lengths.append(2.0)
    true_noises.append(np.linspace(.1,.2,M))
    
    return true_Kfs,true_noises,true_lengths
    
    
def sim_dataset(rs, num_encs,M,n_covs,n_meds,pos_class_rate = 0.5,trainfrac=0.2):
    """
    Returns everything we need to run the model.
    
    Each simulated patient encounter consists of:
        Multivariate time series (labs/vitals)
        Static baseline covariates
        Medication administration times (no values; just a point process)
    """
    true_Kfs,true_noises,true_lengths = gen_MGP_params(M, rs)
        
    end_times = np.random.uniform(10,50,num_encs) #last observation time of the encounter
    num_obs_times = np.random.poisson(end_times,num_encs)+3 #number of observation time points per encounter, increase with  longer series 
    num_obs_values = np.array(num_obs_times*M*trainfrac,dtype="int")
    #number of inputs to RNN. will be a grid on integers, starting at 0 and ending at next integer after end_time
    num_tcn_grid_times = np.array(np.floor(end_times)+1,dtype="int") 
    tcn_grid_times = []
    labels = rs.binomial(1,pos_class_rate,num_encs)                      
    
    T = [];  #actual observation times
    Y = []; ind_kf = []; ind_kt = [] #actual data; indices pointing to which lab, which time
    baseline_covs = np.zeros((num_encs,n_covs)) 
    #each contains an array of size num_tcn_grid_times x n_meds 
    #   simulates a matrix of indicators, where each tells which meds have been given between the
    #   previous grid time and the current.  in practice you will have actual medication administration 
    #   times and will need to convert to this form, for feeding into the RNN
    meds_on_grid = [] 

    print('Simming data...')
    for i in range(num_encs):
        if i%500==0:
            print('%d/%d' %(i,num_encs))
        obs_times = np.insert(np.sort(np.random.uniform(0,end_times[i],num_obs_times[i]-1)),0,0)
        T.append(obs_times)
        l = labels[i]
        y_i,ind_kf_i,ind_kt_i = sim_multitask_GP(obs_times,true_lengths[l],true_noises[l],true_Kfs[l],trainfrac)
        Y.append(y_i); ind_kf.append(ind_kf_i); ind_kt.append(ind_kt_i)
        tcn_grid_times.append(np.arange(num_tcn_grid_times[i]))
        if l==0: #sim some different baseline covs; meds for 2 classes
            baseline_covs[i,:int(n_covs/2)] = rs.normal(0.1,1.0,int(n_covs/2))
            baseline_covs[i,int(n_covs/2):] = rs.binomial(1,0.2,int(n_covs/2))
            meds = rs.binomial(1,.02,(num_tcn_grid_times[i],n_meds))
        else:
            baseline_covs[i,:int(n_covs/2)] = rs.normal(0.2,1.0,int(n_covs/2))
            baseline_covs[i,int(n_covs/2):] = rs.binomial(1,0.1,int(n_covs/2))
            meds = rs.binomial(1,.04,(num_tcn_grid_times[i],n_meds))
        meds_on_grid.append(meds)
    
    T = np.array(T)
    Y = np.array(Y); ind_kf = np.array(ind_kf); ind_kt = np.array(ind_kt)
    meds_on_grid = np.array(meds_on_grid)
    tcn_grid_times = np.array(tcn_grid_times)
    
    return (num_obs_times,num_obs_values,num_tcn_grid_times,tcn_grid_times,
            labels,T,Y,ind_kf,ind_kt,meds_on_grid,baseline_covs)
    
def sim_multitask_GP(times,length,noise_vars,K_f,trainfrac):
    """
    draw from a multitask GP.  
    
    we continue to assume for now that the dim of the input space is 1, ie just time

    M: number of tasks (labs/vitals/time series)
    
    train_frac: proportion of full M x N data matrix Y to include

    """
    M = np.shape(K_f)[0]
    N = len(times)
    n = N*M
    K_t = OU_kernel_np(length,times) #just a correlation function
    Sigma = np.diag(noise_vars)

    K = np.kron(K_f,K_t) + np.kron(Sigma,np.eye(N)) + 1e-6*np.eye(n)
    L_K = np.linalg.cholesky(K)
    
    y = np.dot(L_K,np.random.normal(0,1,n)) #Draw normal
    
    #get indices of which time series and which time point, for each element in y
    ind_kf = np.tile(np.arange(M),(N,1)).flatten('F') #vec by column
    ind_kx = np.tile(np.arange(N),(M,1)).flatten()
               
    #randomly dropout some fraction of fully observed time series
    perm = np.random.permutation(n)
    n_train = int(trainfrac*n)
    train_inds = perm[:n_train]
    
    y_ = y[train_inds]
    ind_kf_ = ind_kf[train_inds]
    ind_kx_ = ind_kx[train_inds]
    
    return y_,ind_kf_,ind_kx_

def OU_kernel_np(length,x):
    """ just a correlation function, for identifiability 
    """
    x1 = np.reshape(x,[-1,1]) #colvec
    x2 = np.reshape(x,[1,-1]) #rowvec
    K_xx = np.exp(-np.abs(x1-x2)/length)    
    return K_xx


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
        num_tcn_grid_times = []
        tcn_grid_times = []
        for i in range(len(times)):
            times[i] = times[i]-min(times[i])
            end_time = times[i][-1]
            num_tcn_grid_time = int(np.floor(end_time)+1)
            num_tcn_grid_times.append(num_tcn_grid_time)
            tcn_grid_times.append(np.arange(num_tcn_grid_time))
        dataset[1] = times
        dataset[5] = np.array(num_tcn_grid_times)
        dataset[6] = np.array(tcn_grid_times)

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
        Xi: grid points (new times for tcn)
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
        #Exp2: increase noise on Sigma 1e-6 to 1e-3, to 1e-1?
        #Sigma = tf.cast(Sigma, tf.float64) ## Experiment: is chol instable and needs float64? Will this crash Memory?
        #draw = Mu + tf.matmul(tf.cast(tf.cholesky(Sigma),tf.float32),xi) 
        draw = Mu + tf.matmul(tf.cholesky(Sigma),xi) 
        draw_reshape = tf.transpose(tf.reshape(tf.transpose(draw),[n_mc_smps,M,nx]),perm=[0,2,1])
        #print('cholesky draw:')
        #print(sess.run(draw_reshape))

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

    return draw_reshape    

@ex.capture        
def get_GP_samples(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
                   num_tcn_grid_times, cov_grid, input_dim,method, gp_params, lab_vitals_only, pad_before): ##,med_cov_grid
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
        Xi = tf.reshape(tf.slice(X,[i,0],[1,num_tcn_grid_times[i]]),[-1])
        X_len = num_tcn_grid_times[i]
                
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

@ex.capture
def get_preds(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
              num_tcn_grid_times, cov_grid, input_dim,method, gp_params, tcn, is_training, n_classes, lab_vitals_only, pad_before, losstype): #med_cov_grid
    """
    helper function. takes in (padded) raw datas, samples MGP for each observation, 
    then feeds it all through the TCN to get predictions.
    
    inputs:
        Y: array of observation values (labs/vitals). batchsize x batch_maxlen_y
        T: array of observation times (times during encounter). batchsize x batch_maxlen_t
        X: array of grid points. batchsize x batch_maxgridlen 
        ind_kf: indiceste into each row of Y, pointing towards which lab/vital. same size as Y
        ind_kt: indices into each row of Y, pointing towards which time. same size as Y
        num_obs_times: number of times observed for each encounter; how long each row of T really is
        num_obs_values: number of lab values observed per encounter; how long each row of Y really is
        num_tcn_grid_times: length of even spaced RNN grid per encounter
                    
    returns:
        predictions (unnormalized log probabilities) for each MC sample of each obs
    """
    
    Z = get_GP_samples(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
                       num_tcn_grid_times, cov_grid, input_dim, method=method, gp_params=gp_params, lab_vitals_only=lab_vitals_only, pad_before=pad_before) #batchsize*num_MC x batch_maxseqlen x num_inputs    ##,med_cov_grid
    Z.set_shape([None,None,input_dim]) #somehow lost shape info, but need this
    N = tf.shape(T)[0] #number of observations 
    
    tcn_logits = tf.layers.dense(
        tcn(Z, training=is_training)[:, -1, :],
        n_classes, activation=None, 
        kernel_initializer=tf.orthogonal_initializer(),
        name='last_linear', reuse=(losstype == 'average') 
    ) # reuse should be true if losstype is average

    '''#duplicate each entry of seqlens, to account for multiple MC samples per observation 
    seqlen_dupe = tf.reshape(tf.tile(tf.expand_dims(num_tcn_grid_times,1),[1,gp_params.n_mc_smps]),[N*gp_params.n_mc_smps])

    #with tf.variable_scope("",reuse=True):
    outputs, states = tf.nn.dynamic_tcn(cell=stacked_lstm,inputs=Z,
                                            dtype=tf.float32,
                                            sequence_length=seqlen_dupe)
    
    final_outputs = states[n_layers-1][1]
    preds =  tf.matmul(final_outputs, out_weights) + out_biases  
    '''
    #pass Z forward through tcn:

    return tcn_logits

@ex.capture
def get_losses(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
              num_tcn_grid_times, cov_grid, input_dim,method, gp_params, tcn, is_training, n_classes, lab_vitals_only, pad_before,
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
        num_tcn_grid_times: length of even spaced RNN grid per encounter
                    
    returns:
        predictions (unnormalized log probabilities) for each MC sample of each obs
    """
    
    Z = get_GP_samples(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
                       num_tcn_grid_times, cov_grid, input_dim, method=method, gp_params=gp_params, lab_vitals_only=lab_vitals_only, pad_before=pad_before) #batchsize*num_MC x batch_maxseqlen x num_inputs    ##,med_cov_grid
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
def mpg_tcn_config():
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



@ex.main
def fit_mgp_tcn(decomposition_method, add_diag, losstype, n_hidden, levels, kernel_size, n_mc_smps, dropout, reduction_dim, batch_size, learning_rate, decay_learning_rate,
                training_iters, time_reset, l1_filter_reg, drop_first_res, L1_penalty, L2_penalty, pad_before, _rnd, _seed, _run, dataset):
    #Parameters (hard-coded for prototyping)
    if len(_run.observers) > 0:
         checkpoint_path = os.path.join(_run.observers[0].dir, 'model_checkpoints')
    else:
        checkpoint_path = 'model_checkpoints'

    #rs = _rnd #fixed seed in np
    #seed = 8675309
    rs = np.random.RandomState(_seed)  #using _seed made name error inside dataset sim function
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Simming dataset:
    num_encs=5000
    M=10
    n_covs=0
    n_meds=0
    
    (num_obs_times,num_obs_values,num_tcn_grid_times,tcn_grid_times,labels,times,
       values,ind_lvs,ind_times,meds_on_grid,covs) = sim_dataset(rs, num_encs,M,n_covs,n_meds)
        
    N_tot = len(labels) #total encounters
    
    train_test_perm = rs.permutation(N_tot)
    val_frac = 0.1 #fraction of full data to set aside for testing
    va_ind = train_test_perm[:int(val_frac*N_tot)]
    tr_ind = train_test_perm[int(val_frac*N_tot):]
    Nva = len(va_ind); Ntr = len(tr_ind)
    
    #Break everything out into train/test
    covs_tr = covs[tr_ind,:]; covs_va = covs[va_ind,:]
    labels_tr = labels[tr_ind]; labels_va = labels[va_ind]
    times_tr = times[tr_ind]; times_va = times[va_ind]
    values_tr = values[tr_ind]; values_va = values[va_ind]
    ind_lvs_tr = ind_lvs[tr_ind]; ind_lvs_va = ind_lvs[va_ind]
    ind_times_tr = ind_times[tr_ind]; ind_times_va = ind_times[va_ind]
    num_obs_times_tr = num_obs_times[tr_ind]; num_obs_times_va = num_obs_times[va_ind]
    num_obs_values_tr = num_obs_values[tr_ind]; num_obs_values_va = num_obs_values[va_ind]
    tcn_grid_times_tr = tcn_grid_times[tr_ind]; tcn_grid_times_va = tcn_grid_times[va_ind]     
    num_tcn_grid_times_tr = num_tcn_grid_times[tr_ind]; num_tcn_grid_times_va = num_tcn_grid_times[va_ind]  

    # Load dataset
    #train_data, validation_data, test_data, variables = get_dataset()
    ''' #here, we don't load real data, but for quick test simulate data
    full_dataset = get_dataset() # load dataset
    '''
    data_sources = dataset['data_sources'] #get names of data sources (labs, vitals ,..)
    lab_vitals_only = dataset['lab_vitals_only']
    num_obs_thres = dataset['num_obs_thres']
    min_pad_length = dataset['min_pad_length'] #minimal time series length a batch should be padded to (for TCNs..)
    obs_min = np.max([10, n_mc_smps]) #remove the samples with less than 10 observation values, this lanczos impl fails when mc_smps > num_obs
    method = DecompositionMethod(decomposition_method, add_diag)

    #Get class imbalance (for weighted loss):
    case_prev = labels_tr.sum()/float(len(labels_tr)) #get prevalence of cases in train dataset
    class_imb = 1/case_prev #class imbalance to use as class weight if losstype='weighted'
    
    print("data fully setup!")    
    sys.stdout.flush()

    #####
    ##### Setup model and graph
    ##### 
    
    # Learning Parameters
    
    #learning_rate = 0.01 #NOTE may need to play around with this or decay it
    # Adaptive learning rate:
    
    decay_step = int(Ntr/batch_size) #100 #after how many batches will the learning rate be adjusted..
    
    ##test_freq     = Ntr/batch_size #eval on test set after this many batches
    test_freq = int(Ntr/batch_size / 4)
    # Network Parameters
    n_classes = 2 #binary outcome
    
    if lab_vitals_only:
        input_dim = M
    elif 'covs' in data_sources:
        input_dim = M + n_covs #dimensionality of input sequence. ##
    else:
        input_dim = M 

    ##### Create graph
    
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
    num_tcn_grid_times = tf.placeholder(tf.int32, [None]) #length of each grid to be fed into RNN in batch
    
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
    #initialize tcn inputs:
    # Z = tf.placeholder("float", [None,None,None]) #GP_draws: [batch_size*n_mc_smps, batch_max_length, M]
    is_training = tf.placeholder("bool")
    #define tcn outputs:

    ##### Get predictions and feed into optimization
    if losstype=='average':
        losses = get_losses(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,num_tcn_grid_times, cov_grid, input_dim,
                      method=method, gp_params=gp_params, tcn=tcn, is_training=is_training, n_classes=n_classes, lab_vitals_only=lab_vitals_only, pad_before=pad_before,
                      labels=O_dupe_onehot, pos_weight=class_imb)
        loss_fit = tf.reduce_sum(losses)
    
    preds = get_preds(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,num_tcn_grid_times, cov_grid, input_dim,
                      method=method, gp_params=gp_params, tcn=tcn, is_training=is_training, n_classes=n_classes, lab_vitals_only=lab_vitals_only, pad_before=pad_before, losstype=losstype)    #med_cov_grid
    probs,accuracy = get_probs_and_accuracy(preds,O)
    
    # Define optimization problem
    #if losstype=='regular':
    #   loss_fit = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=preds,labels=O_dupe_onehot))
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

    count_parameters()
    
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
        tcn_grid_times_va, covs_va, num_tcn_grid_times_va, min_pad_length)

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
                    tcn_grid_times_tr[inds], covs_tr[inds,:], num_tcn_grid_times_tr[inds], min_pad_length) ## meds_on_grid_tr[inds],covs_tr[inds,:]

            feed_dict={Y:Y_pad,T:T_pad,ind_kf:ind_kf_pad,ind_kt:ind_kt_pad,X:X_pad, cov_grid:cov_pad,
               num_obs_times:num_obs_times_tr[inds],
               num_obs_values:num_obs_values_tr[inds],
               num_tcn_grid_times:num_tcn_grid_times_tr[inds],O:labels_tr[inds], is_training: True} ##med_cov_grid:meds_cov_pad,                        

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
                val_t = time()
                #Batch-wise Validation Phase (more memory saving)
                va_probs_tot = np.array([])
                va_perm = rs.permutation(Nva)
                va_labels_tot = labels_va[va_perm]
                for v_s,v_e in zip(va_starts,va_ends):
                    va_inds = va_perm[v_s:v_e]

                    va_feed_dict={Y:Y_pad_va[va_inds,:], T:T_pad_va[va_inds,:], ind_kf:ind_kf_pad_va[va_inds,:], 
                                ind_kt:ind_kt_pad_va[va_inds,:], X:X_pad_va[va_inds,:],
                                cov_grid:cov_pad_va[va_inds,:,:], num_obs_times:num_obs_times_va[va_inds],
                                num_obs_values:num_obs_values_va[va_inds], num_tcn_grid_times:num_tcn_grid_times_va[va_inds],
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
        
        ### Takes about ~1-2 secs per batch of 50 at these settings, so a few minutes each epoch
        ### Should converge reasonably quickly on this toy example with these settings in a few epochs
    return {'Best Validation AUPRC': best_val}

if __name__ == '__main__':
    ex.run_commandline()
