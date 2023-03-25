# To run the train.py and train the model, run the following command: CUDA_LAUNCH_BLOCKING=1 python train.py
# Change path at line 39, 183 and 235 to local corresponding path
# Model type can be set by changing variable at line 30 - available model in the README file
# To select how the data is split, change variable 'split_mode' at line 58 to 'user' if split by user, or 'temporal' if split by days
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from collections import deque, Counter
from sklearn import metrics as skmetrics
import pickle
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from model import RnnParameterData
from model import AppLocUserLoss
from model import AppPreLocPreUserIdenGtrLinear, AppPreLocPreUserIdenGtr, AppPreLocPreUserIdenPOIGtr, AppPreLocPreUserIdenPOIFCGtr
from model import AppPreLocPreUserIdenPOIWeatherGtr, AppPreLocPreUserIdenPOIWeatherFCGtr
from helper import generate_input, run_simple

# Choose the model for training - find other available model in README file
model_type = 'AppPreLocPreUserIdenPOIWeatherGtr'

# Determine whether or not GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Initiate the parameters
data = {}
path = './DeepAppmodel/data/user_7_days'

for u in os.listdir(path):

    uid = int(u[:-7])
    file_name = '/'.join([path, u])

    try:
        with open(file_name, 'rb') as f:
            dic = pickle.load(f)
            data[uid] = dic[uid]
    except:
        print(dic)

# Initiate the parameters
parameters = RnnParameterData(poi_emb_size = 4, loc_emb_size=256, uid_emb_size=64, tim_emb_size=16, app_size=2000, app_encoder_size=512, hidden_size=128,acc_threshold=0.3,
                lr=1e-4, lr_step=2, lr_decay=0.1, dropout_p=0.5, L2=1e-5, clip=5.0, optim='Adam', lr_schedule='Loss',
                history_mode='avg', attn_type='general', model_mode=model_type, top_size=16, rnn_type = 'GRU', loss_alpha =0.2,loss_beta =0.2,
                threshold=0.05, epoch_max=100, app_emb_mode ='sum',
                baseline_mode=None, loc_emb_negative=5, loc_emb_batchsize=256, input_mode='short',test_start = 2, split_mode='user')

"""metric"""
T = 10
T0 = T
lr = parameters.lr
metrics = {'train_loss': [], 'valid_loss': [], 'avg_app_auc': [], 'avg_app_map': [], 'avg_app_precision': [], 'avg_app_recall': [], 'avg_loc_top1': [], 'avg_loc_top5': [],'avg_loc_top10': [], 'avg_uid_top1': [], 'avg_uid_top10': [], 'valid_acc': {}}

if parameters.baseline_mode == None:			
    print("================Run models=============")
    print(parameters.model_mode)
    # parameters.uid_size = n_users
    #Model Training
    print('Split training and testing data', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # if 'Topk' in parameters.model_mode:
    # 	print('using topk model')
    # 	user_topk = generate_input_topk(parameters,'train',loc_old2newid, mode2=None, candidate=candidate)
    # else:
    # 	user_topk = None
    if parameters.input_mode == 'short':
        data_train, train_idx = generate_input(data, 'train', parameters.split_mode)
        data_test, test_idx = generate_input(data, 'val', parameters.split_mode)
    # elif parameters.input_mode == 'short_history':
    # 	data_train, train_idx = generate_input_history(parameters, 'train', loc_old2newid, user_topk, mode2=parameters.history_mode,candidate=candidate)
    # 	data_test, test_idx = generate_input_history(parameters, 'test', loc_old2newid, user_topk, mode2=parameters.history_mode,candidate=candidate)
    # elif parameters.input_mode == 'long':
    # 	data_train, train_idx = generate_input_long_history(parameters, 'train', loc_old2newid, user_topk, mode2=parameters.history_mode,candidate=candidate)
    # 	data_test, test_idx = generate_input_long_history(parameters, 'test', loc_old2newid, user_topk, mode2=parameters.history_mode,candidate=candidate)

    
    """Model Init"""
    print('Model Init!', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    for mi in range(1):
        if parameters.model_mode in ['AppPreLocPreUserIdenGt','AppPreLocPreUserIdenGtr']:
            model = AppPreLocPreUserIdenGtr(parameters=parameters).to(device)
            criterion = AppLocUserLoss(parameters=parameters).to(device)
        elif parameters.model_mode in ['AppPreLocPreUserIdenGtrLinear']:
            model = AppPreLocPreUserIdenGtrLinear(parameters=parameters).to(device)
            criterion = AppLocUserLoss(parameters=parameters).to(device)
        elif parameters.model_mode in ['AppPreLocPreUserIdenPOIGtr',]:
            model = AppPreLocPreUserIdenPOIGtr(parameters=parameters).to(device)
            criterion = AppLocUserLoss(parameters=parameters).to(device)
        elif parameters.model_mode in ['AppPreLocPreUserIdenPOIFCGtr']:
            model = AppPreLocPreUserIdenPOIFCGtr(parameters=parameters).to(device)
            criterion = AppLocUserLoss(parameters=parameters).to(device)
        elif parameters.model_mode in ['AppPreLocPreUserIdenPOIWeatherGtr']:
            model = AppPreLocPreUserIdenPOIWeatherGtr(parameters=parameters).to(device)
            criterion = AppLocUserLoss(parameters=parameters).to(device)
        elif parameters.model_mode in ['AppPreLocPreUserIdenPOIWeatherFCGtr']:
            model = AppPreLocPreUserIdenPOIWeatherFCGtr(parameters=parameters).to(device)
            criterion = AppLocUserLoss(parameters=parameters).to(device)

    #Forward network with randomly initialization
    for epoch in range(1):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=parameters.L2)

        # ReduceLROnPlateau look at the a metrics and reduce the learning rate when it does not improve
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=parameters.lr_step,factor=parameters.lr_decay, threshold=1e-3)

        # Set mode to 'train_test' which look at train dataset with test model without calculating the gradient
        print('========================>Epoch:{:0>2d} lr:{}<=================='.format(epoch,lr))
        model, avg_train_loss, prediction = run_simple(data_train, train_idx, 'train_test', parameters.input_mode, lr, parameters.clip, model, optimizer, criterion, parameters.model_mode, parameters.test_start, parameters.acc_threshold)
        print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_train_loss, lr))
        metrics['train_loss'].append(avg_train_loss)
        print('-'*30)

        # Set mode to test mode to get the loss and metrics from the test set
        avg_loss, avg_acc, users_acc, prediction = run_simple(data_test, test_idx, 'test', parameters.input_mode, lr, parameters.clip, model,optimizer, criterion, parameters.model_mode, parameters.test_start, parameters.acc_threshold)
        print('==>Test Loss:{:.4f}'.format(avg_loss))
        print('==>Test Acc App_AUC:{:.4f}   App_map:{:.4f}    App_Precision:{:.4f}   App_Recall:{:.4f} App_F1:{:.4f}'.format(avg_acc['app_auc'], avg_acc['app_map'], avg_acc['app_precision'], avg_acc['app_recall'], avg_acc['app_f1']))
        # print('            Loc_top1:{:.4f}  Loc_top5:{:.4f}  Loc_top10:{:.4f}'.format(avg_acc['loc_top1'],avg_acc['loc_top5'], avg_acc['loc_top10']))
        # print('            Uid_top1:{:.4f}  Uid_top10:{:.4f}'.format(avg_acc['uid_top1'], avg_acc['uid_top10']))
        # print('            Loc_top10:{:.4f}'.format(avg_acc['loc_top10']))
        # print('            Uid_top10:{:.4f}'.format(avg_acc['uid_top10']))
        print('-'*30)

        # Saving the loss and metrics in this epoch
        metrics['valid_loss'].append(avg_loss) #total average loss
        metrics['valid_acc'][epoch] = users_acc #accuracy for each user
        metrics['avg_app_auc'].append(0) #total average accuracy
        metrics['avg_app_map'].append(0)
        metrics['avg_app_precision'].append(0)
        metrics['avg_app_recall'].append(0)
        # metrics['avg_loc_top1'].append(0)
        # metrics['avg_loc_top5'].append(0)
        # metrics['avg_loc_top10'].append(0)
        # metrics['avg_uid_top1'].append(0)
        # metrics['avg_uid_top10'].append(0)
        #prediction_all[epoch]['test'] = prediction

    st = datetime.now()
    start_time = datetime.now()
    for epoch in range(1, parameters.epoch):
        
        print('========================>Epoch:{:0>2d} lr:{}<=================='.format(epoch,lr))
        model, avg_train_loss, prediction = run_simple(data_train, train_idx, 'train', parameters.input_mode, lr, parameters.clip, model, optimizer, criterion, parameters.model_mode, parameters.test_start, parameters.acc_threshold)
        print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_train_loss, lr))
        metrics['train_loss'].append(avg_train_loss)
        print('-'*30)

        avg_loss, avg_acc, users_acc, prediction = run_simple(data_test, test_idx, 'test', parameters.input_mode, lr, parameters.clip, model,optimizer, criterion, parameters.model_mode, parameters.test_start, parameters.acc_threshold)
        print('==>Test Loss:{:.4f}'.format(avg_loss))
        print('==>Test Acc App_AUC:{:.4f}   App_map:{:.4f}    App_Precision:{:.4f}   App_Recall:{:.4f} App_F1:{:.4f}'.format(avg_acc['app_auc'], avg_acc['app_map'], avg_acc['app_precision'], avg_acc['app_recall'], avg_acc['app_f1']))
        # print('            Loc_top1:{:.4f}  Loc_top5:{:.4f}  Loc_top10:{:.4f}'.format(avg_acc['loc_top1'],avg_acc['loc_top5'], avg_acc['loc_top10']))
        # print('            Uid_top1:{:.4f}  Uid_top10:{:.4f}'.format(avg_acc['uid_top1'], avg_acc['uid_top10']))
        # print('            Loc_top10:{:.4f}'.format(avg_acc['loc_top10']))
        # print('            Uid_top10:{:.4f}'.format(avg_acc['uid_top10']))
        print('-'*30)

        metrics['valid_loss'].append(avg_loss) #total average loss
        metrics['valid_acc'][epoch] = users_acc #accuracy for each user
        metrics['avg_app_auc'].append(avg_acc['app_auc']) #total average accuracy
        metrics['avg_app_map'].append(avg_acc['app_map'])
        metrics['avg_app_precision'].append(avg_acc['app_precision'])
        metrics['avg_app_recall'].append(avg_acc['app_recall'])
        # metrics['avg_loc_top1'].append(avg_acc['loc_top1'])
        # metrics['avg_loc_top5'].append(avg_acc['loc_top5'])
        # metrics['avg_loc_top10'].append(avg_acc['loc_top10'])
        # metrics['avg_uid_top1'].append(avg_acc['uid_top1'])
        # metrics['avg_uid_top10'].append(avg_acc['uid_top10'])
        #prediction_all[epoch]['test'] = prediction

        save_name_tmp = parameters.model_mode + str(start_time)
        
        torch.save({'model_state_dict': model.state_dict()},
                    './DeepAppmodel/model/{}'.format(save_name_tmp))

        # They use the ReduceLROnPlateau scheduler which reduce the learning rate when the metrics does not improve
        if parameters.lr_schedule == 'Loss':

            # Since we have AppPre, the scheduler will look at the app_map (mean average precision anyway)
            if 'AppPre' in parameters.model_mode:
                scheduler.step(avg_acc['app_map'])
            elif 'LocPre' in parameters.model_mode:
                scheduler.step(avg_acc['loc_top1'])
            elif 'UserIden' in parameters.model_mode:
                scheduler.step(avg_acc['uid_top1'])
            lr_last = lr

            lr = optimizer.param_groups[0]['lr'] #### Have not figured out what is this

            # They save a new model everytime the lr reduces
            if lr_last > lr:
                if 'AppPre' in parameters.model_mode:
                    load_epoch = np.argmax(metrics['avg_app_map'])
                elif 'LocPre' in parameters.model_mode:
                    load_epoch = np.argmax(metrics['avg_loc_top1'])
                else:
                    load_epoch = np.argmax(metrics['avg_uid_top1'])          

                load_name_tmp = 'ep_' + str(load_epoch) + '_' + str(start_time) + '.m'


        # Print the required time to run epoch including calculating the metrics
        if epoch == 1:
            print('single epoch time cost:{}'.format(datetime.now() - start_time))
        
        # Early-stopping if the learning rate is smaller than the value
        if lr <= 0.9 * 1e-6:
            break
        print('='*30)

    
    # Print out the stats of the best epoch which the model should have been saved
    overhead = datetime.now() - start_time
    if 'AppPre' in parameters.model_mode:
        load_epoch = np.argmax(metrics['avg_app_map'])
        print('==>Test Best Epoch:{:0>2d}   App_AUC:{:.4f}   app_map:{:.4f}   App_Precision:{:.4f}   App_Recall:{:.4f} '.format(load_epoch, metrics['avg_app_auc'][load_epoch], metrics['avg_app_map'][load_epoch], metrics['avg_app_precision'][load_epoch], metrics['avg_app_recall'][load_epoch]))
    # elif 'LocPre' in parameters.model_mode: 
    #     load_epoch = np.argmax(metrics['avg_loc_top1'])
    #     print('==>Test Best Epoch:{:0>2d}   Loc_Top1:{:.4f}   Loc_top10:{:.4f}'.format(load_epoch, metrics['avg_loc_top1'][load_epoch], metrics['avg_loc_top10'][load_epoch]))
    # else:
    #     load_epoch = np.argmax(metrics['avg_uid_top1'])
    #     print('==>Test Best Epoch:{:0>2d}   Uid_Top1:{:.4f}   Uid_top10:{:.4f}'.format(load_epoch, metrics['avg_uid_top1'][load_epoch], metrics['avg_uid_top10'][load_epoch]))
    

    # We have the training and validation results
    with open('./DeepAppmodel/results/{}_graph.pickle'.format(save_name_tmp), 'wb') as m:
        pickle.dump(metrics, m, protocol=pickle.HIGHEST_PROTOCOL)