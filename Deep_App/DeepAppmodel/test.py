# To run the model and predict on the test set, run the following command: CUDA_LAUNCH_BLOCKING=1 python test.py
# Change path at line 32 and 40 to local corresponding path
# Model type has to be set accordingly to the best model by path set on line 32
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
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from model import RnnParameterData
from model import AppLocUserLoss
from model import AppPreLocPreUserIdenGtrLinear, AppPreLocPreUserIdenGtr, AppPreLocPreUserIdenPOIGtr, AppPreLocPreUserIdenPOIFCGtr
from model import AppPreLocPreUserIdenPOIWeatherGtr, AppPreLocPreUserIdenPOIWeatherFCGtr
from helper import generate_input, run_simple

# Choose the model file for testing
model_type = 'AppPreLocPreUserIdenPOIWeatherGtr'

# AppPreLocPreUserIdenPOIWeatherGtronlyapp10 is the best model by the team
model_path = './DeepAppmodel/model/AppPreLocPreUserIdenPOIWeatherGtronlyapp10'


# Determine whether or not GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

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
lr = parameters.lr
metrics = {'train_loss': [], 'valid_loss': [], 'avg_app_auc': [], 'avg_app_map': [], 'avg_app_precision': [], 'avg_app_recall': [], 'avg_loc_top1': [], 'avg_loc_top5': [],'avg_loc_top10': [], 'avg_uid_top1': [], 'avg_uid_top10': [], 'valid_acc': {}}

if parameters.baseline_mode == None:			
	print("================Run models=============")
	print('Split training and testing data', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
	if parameters.input_mode == 'short':
		if parameters.split_mode == 'temporal':
			data_test, test_idx = generate_input(data, 'test', 'temporal') #generate input for test if split mode is temporal split (by days)
		elif parameters.split_mode == 'user':
			data_test, test_idx = generate_input(data, 'test', 'user') #generate input for test if split mode is user split
		else:
			raise ValueError('Invalid split mode')
		

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

	"""Load model"""

	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=parameters.L2)

	# ReduceLROnPlateau look at the a metrics and reduce the learning rate when it does not improve
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=parameters.lr_step,factor=parameters.lr_decay, threshold=1e-3)

	model.load_state_dict(torch.load(model_path)["model_state_dict"])
	model = model.to(device)
	model.eval()


	# Set mode to test mode to get the loss and metrics from the test set
	avg_loss, avg_acc, users_acc, prediction = run_simple(data_test, test_idx, 'test', parameters.input_mode, lr, parameters.clip, model,optimizer, criterion, parameters.model_mode, parameters.test_start, parameters.acc_threshold)

	print('==>Test Loss:{:.4f}'.format(avg_loss))
	print('==>Test Acc App_AUC:{:.4f}   App_map:{:.4f}    App_Precision:{:.4f}   App_Recall:{:.4f} App_F1:{:.4f}'.format(avg_acc['app_auc'], avg_acc['app_map'], avg_acc['app_precision'], avg_acc['app_recall'], avg_acc['app_f1']))
	print('-'*30)

	# Saving the loss and metrics in this epoch
	metrics['valid_loss'].append(avg_loss) #total average loss
	metrics['valid_acc'][0] = users_acc #accuracy for each user
	metrics['avg_app_auc'].append(0) #total average accuracy
	metrics['avg_app_map'].append(0)
	metrics['avg_app_precision'].append(0)
	metrics['avg_app_recall'].append(0)

	st = datetime.now()
	start_time = datetime.now()