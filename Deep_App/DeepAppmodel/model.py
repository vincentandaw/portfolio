# coding: utf-8
from __future__ import print_function
from __future__ import division

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
from torch.autograd import Variable
from datetime import datetime

from sklearn.exceptions import UndefinedMetricWarning

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Use GPU is there is any
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# #############  DeepApp Model ####################### #
class AppPreLocPreUserIdenGtr(nn.Module):
	"""baseline rnn model, location prediction """
	def __init__(self, parameters):
		super(AppPreLocPreUserIdenGtr, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.uid_size = parameters.uid_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)
		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size + self.loc_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.fc_uid = nn.Linear(self.hidden_size, self.uid_size)
		self.dec_app = nn.Linear(self.hidden_size, self.app_size)
		self.fc_loc = nn.Linear(self.hidden_size, self.loc_size)

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform_(t)
		for t in hh:
			nn.init.orthogonal_(t)
		for t in b:
			nn.init.constant_(t, 0)
			
	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.to(device)
			c1 = c1.to(device)
			app_emb = app_emb.to(device)

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		loc_emb = self.emb_loc(loc)

		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		elif self.app_emd_mode == 'avg':
			app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb.unsqueeze(1)

		x = torch.cat((tim_emb, app_emb, ptim_emb, loc_emb), 2)
		x = self.dropout(x)

		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)
		
		app_out = self.dec_app(out)
		app_score = torch.sigmoid(app_out)
		
		loc_out = self.fc_loc(out)
		loc_score = F.log_softmax(loc_out, dim=0)  # calculate loss by NLLoss

		user_out = self.fc_uid(out)
		user_score = F.log_softmax(user_out, dim=0)

		return app_score,loc_score,user_score


# #############  GRU with POI ####################### #
class AppPreLocPreUserIdenPOIGtr(nn.Module):
	"""baseline rnn model, location prediction """
	def __init__(self, parameters):
		super(AppPreLocPreUserIdenPOIGtr, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.uid_size = parameters.uid_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode
		self.poi_size = parameters.poi_size
		self.poi_emb_size = parameters.poi_emb_size
		self.emb_poi = nn.Embedding(self.poi_size, self.poi_emb_size)
		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)
		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size + self.loc_emb_size + self.poi_emb_size

		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.fc_uid = nn.Linear(self.hidden_size, self.uid_size)
		self.dec_app = nn.Linear(self.hidden_size, self.app_size)
		self.fc_loc = nn.Linear(self.hidden_size, self.loc_size)
		

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform_(t)
		for t in hh:
			nn.init.orthogonal_(t)
		for t in b:
			nn.init.constant_(t, 0)
			
	def forward(self, tim, app, loc, uid, ptim, poi):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		poi_emb = Variable(torch.zeros(len(poi), self.poi_emb_size))
		
		if self.use_cuda:
			h1 = h1.to(device)
			c1 = c1.to(device)
			app_emb = app_emb.to(device)

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		loc_emb = self.emb_loc(loc)

		if self.app_emd_mode == 'sum':
			poi_emb = poi.mm(self.emb_poi.weight)
			app_emb = app.mm(self.emb_app.weight)

		app_emb = app_emb.unsqueeze(1)
		poi_emb = poi_emb.unsqueeze(1)

		x = torch.cat((tim_emb, app_emb, ptim_emb, loc_emb, poi_emb), 2)
		x = self.dropout(x)

		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)
		
		app_out = self.dec_app(out)
		app_score = torch.sigmoid(app_out)
		
		loc_out = self.fc_loc(out)
		loc_score = F.log_softmax(loc_out, dim=0)  # calculate loss by NLLoss

		user_out = self.fc_uid(out)
		user_score = F.log_softmax(user_out, dim=0)

		return app_score,loc_score,user_score


# #############  FC with POI ####################### #
class AppPreLocPreUserIdenPOIFCGtr(nn.Module):
	"""baseline rnn model, location prediction """
	def __init__(self, parameters):
		super(AppPreLocPreUserIdenPOIFCGtr, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.uid_size = parameters.uid_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode
		self.poi_size = parameters.poi_size
		self.poi_emb_size = parameters.poi_emb_size
		self.emb_poi = nn.Embedding(self.poi_size, self.poi_emb_size)
		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)
		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size + self.loc_emb_size + self.poi_emb_size
		self.fc_middle = nn.Linear(input_size, self.hidden_size)
		self.fc_middle2 = nn.Linear(self.hidden_size, self.hidden_size)
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.fc_uid = nn.Linear(self.hidden_size, self.uid_size)
		self.dec_app = nn.Linear(self.hidden_size, self.app_size)
		self.fc_loc = nn.Linear(self.hidden_size, self.loc_size)
	

	def forward(self, tim, app, loc, uid, ptim, poi):

		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		poi_emb = Variable(torch.zeros(len(poi), self.poi_emb_size))

		if self.use_cuda:
			app_emb = app_emb.to(device)

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		loc_emb = self.emb_loc(loc)

		if self.app_emd_mode == 'sum':
			poi_emb = poi.mm(self.emb_poi.weight)
			app_emb = app.mm(self.emb_app.weight)

		app_emb = app_emb.unsqueeze(1)
		poi_emb = poi_emb.unsqueeze(1)

		x = torch.cat((tim_emb, app_emb, ptim_emb, loc_emb, poi_emb), 2)
		x = x.squeeze(1)
		x = self.dropout(x)
		x = self.fc_middle(x)  # input -> 128
		# x = self.fc_middle2(x)  # 128 -> self.hiddensize

		out = F.selu(x)
		
		app_out = self.dec_app(out)
		app_score = torch.sigmoid(app_out)
		
		loc_out = self.fc_loc(out)
		loc_score = F.log_softmax(loc_out, dim=0)  # calculate loss by NLLoss

		user_out = self.fc_uid(out)
		user_score = F.log_softmax(user_out, dim=0)

		return app_score,loc_score,user_score

# #############  Weather with GRU ####################### #
class AppPreLocPreUserIdenPOIWeatherGtr(nn.Module):
	"""baseline rnn model, location prediction """
	def __init__(self, parameters):
		super(AppPreLocPreUserIdenPOIWeatherGtr, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.uid_size = parameters.uid_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode
		# ADD POI -----------------------------------------------------------------------
		self.poi_size = parameters.poi_size
		self.poi_emb_size = parameters.poi_emb_size
		self.emb_poi = nn.Embedding(self.poi_size, self.poi_emb_size)
		# ADD Weather -------------------------------------------------------------------
		# self.weather_size = parameters.weather_size
		# self.weather_emb_size = parameters.weather_emb_size
		# self.emb_weather = nn.Embedding(self.weather_size, self.weather_emb_size)
		# ------------------------------------------------------------------------------
		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)
		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size + self.loc_emb_size + self.poi_emb_size + 1

		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.fc_uid = nn.Linear(self.hidden_size, self.uid_size)
		self.dec_app = nn.Linear(self.hidden_size, self.app_size)
		self.fc_loc = nn.Linear(self.hidden_size, self.loc_size)
		

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform_(t)
		for t in hh:
			nn.init.orthogonal_(t)
		for t in b:
			nn.init.constant_(t, 0)
			
	def forward(self, tim, app, loc, uid, ptim, poi, weather):

		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		poi_emb = Variable(torch.zeros(len(poi), self.poi_emb_size))

		if self.use_cuda:
			h1 = h1.to(device)
			c1 = c1.to(device)
			app_emb = app_emb.to(device)
			poi_emb = poi_emb.to(device)

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		loc_emb = self.emb_loc(loc)

		if self.app_emd_mode == 'sum':
			poi_emb = poi.mm(self.emb_poi.weight)
			app_emb = app.mm(self.emb_app.weight)
		app_emb = app_emb.unsqueeze(1)
		poi_emb = poi_emb.unsqueeze(1)
		weather_emb = weather.unsqueeze(1).unsqueeze(1)

		x = torch.cat((tim_emb, app_emb, ptim_emb, loc_emb,poi_emb,weather_emb), 2)
		x = self.dropout(x)

		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)
		
		app_out = self.dec_app(out)
		app_score = torch.sigmoid(app_out)
		
		loc_out = self.fc_loc(out)
		loc_score = F.log_softmax(loc_out, dim=0)  # calculate loss by NLLoss

		user_out = self.fc_uid(out)
		user_score = F.log_softmax(user_out, dim=0)

		return app_score,loc_score,user_score

# #############  Weather with FC ####################### #
class AppPreLocPreUserIdenPOIWeatherFCGtr(nn.Module):
	"""baseline rnn model, location prediction """
	def __init__(self, parameters):
		super(AppPreLocPreUserIdenPOIWeatherFCGtr, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.uid_size = parameters.uid_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode
		# ADD POI -----------------------------------------------------------------------
		self.poi_size = parameters.poi_size
		self.poi_emb_size = parameters.poi_emb_size
		self.emb_poi = nn.Embedding(self.poi_size, self.poi_emb_size)
		# ADD Weather -------------------------------------------------------------------
		# self.weather_size = parameters.weather_size
		# self.weather_emb_size = parameters.weather_emb_size
		# self.emb_weather = nn.Embedding(self.weather_size, self.weather_emb_size)
		# ------------------------------------------------------------------------------
		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)
		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size + self.loc_emb_size + self.poi_emb_size + 1

		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.fc_middle = nn.Linear(input_size, self.hidden_size)
		self.fc_uid = nn.Linear(self.hidden_size, self.uid_size)
		self.dec_app = nn.Linear(self.hidden_size, self.app_size)
		self.fc_loc = nn.Linear(self.hidden_size, self.loc_size)
		

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform_(t)
		for t in hh:
			nn.init.orthogonal_(t)
		for t in b:
			nn.init.constant_(t, 0)
			
	def forward(self, tim, app, loc, uid, ptim, poi, weather):

		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		poi_emb = Variable(torch.zeros(len(poi), self.poi_emb_size))

		if self.use_cuda:
			h1 = h1.to(device)
			c1 = c1.to(device)
			app_emb = app_emb.to(device)
			poi_emb = poi_emb.to(device)

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		loc_emb = self.emb_loc(loc)

		if self.app_emd_mode == 'sum':
			poi_emb = poi.mm(self.emb_poi.weight)
			app_emb = app.mm(self.emb_app.weight)
		app_emb = app_emb.unsqueeze(1)
		poi_emb = poi_emb.unsqueeze(1)
		weather_emb = weather.unsqueeze(1).unsqueeze(1)

		x = torch.cat((tim_emb, app_emb, ptim_emb, loc_emb, poi_emb, weather_emb), 2)
		x = x.squeeze(1)
		x = self.dropout(x)
		x = self.fc_middle(x)  # input -> 128
		# x = self.fc_middle2(x)  # 128 -> self.hiddensize

		out = F.selu(x)
		
		app_out = self.dec_app(out)
		app_score = torch.sigmoid(app_out)
		
		loc_out = self.fc_loc(out)
		loc_score = F.log_softmax(loc_out, dim=0)  # calculate loss by NLLoss

		user_out = self.fc_uid(out)
		user_score = F.log_softmax(user_out, dim=0)

		return app_score,loc_score,user_score


class AppPreLocPreUserIdenGtrLinear(nn.Module):
	"""baseline rnn model, location prediction """
	def __init__(self, parameters):
		super(AppPreLocPreUserIdenGtrLinear, self).__init__()
		self.tim_size = parameters.tim_size
		self.app_size = parameters.app_size
		self.loc_size = parameters.loc_size
		self.uid_size = parameters.uid_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda

		input_size = self.tim_size*2 + self.app_size + self.loc_size
		output_size = self.app_size + self.loc_size + self.uid_size
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.fc = nn.Linear(input_size, output_size)

			
	def forward(self, tim, app, loc, uid, ptim):
		x = torch.cat((tim, app, loc, ptim), 1)

		out = self.fc(x)
		out = self.dropout(out)

		out = out.squeeze(1)
		out = F.selu(out)
		
		app_out = out[:,:self.app_size]
		app_score = F.sigmoid(app_out)
		
		loc_out = out[:,self.app_size:self.app_size+self.loc_size]
		loc_score = F.log_softmax(loc_out)  # Cross-entropy loss is equal to (log_softmax then NLLLoss)
		
		user_out = out[:,self.app_size+self.loc_size:]
		user_score = F.log_softmax(user_out)

		return app_score,loc_score,user_score

# ############# Define Loss ####################### #
# USE THIS LOSS
class AppLocUserLoss(nn.Module):
	def __init__(self,parameters):
		super(AppLocUserLoss, self).__init__()
		self.alpha = parameters.loss_alpha
		self.beta = parameters.loss_beta
	def forward(self, app_scores, app_target, loc_scores, loc_target, uid_scores, uid_target):
		app_loss = nn.BCELoss()
		loc_loss = nn.NLLLoss()
		uid_loss = nn.NLLLoss()
		loss_app = app_loss(app_scores, app_target)
		loss_loc = loc_loss(loc_scores, loc_target)
		loss_uid = uid_loss(uid_scores, uid_target)
		# if fake tasks included, add to the first loss_app in return: + self.alpha*loss_loc + self.beta*loss_uid
		return loss_app , loss_app, loss_loc, loss_uid




class RnnParameterData(object):
    def __init__(self, poi_emb_size=17, loc_emb_size=128, uid_emb_size=64, tim_emb_size=16, app_size=20, app_encoder_size=32, hidden_size=128,acc_threshold=0.3,
                    lr=1e-4, lr_step=1, lr_decay=0.1, dropout_p=0.5, L2=1e-5, clip=5.0, optim='Adam', lr_schedule='Loss',
                    history_mode='avg', attn_type='general', model_mode='attn', top_size=16, rnn_type = 'GRU', loss_alpha =0.5,loss_beta =0.01,
                    threshold=0.05, epoch_max=20,users_end=1000, app_emb_mode ='sum',
                    baseline_mode=None, loc_emb_negative=5, loc_emb_batchsize=256, input_mode='short',test_start = 2, split_mode = 'temporal'):

        print('===>data load start!',datetime.now().strftime('%Y-%m-%d %H:%M:%S') )
        print('===>data load complete!',datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        # self.data_neural = data
        self.tim_size = 48
        self.loc_size = 10000
        self.poi_size = 17
        self.uid_size = users_end
        self.app_size = app_size
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.uid_emb_size = uid_emb_size
        self.app_encoder_size = app_encoder_size
        self.hidden_size = hidden_size
        self.top_size = top_size
        self.test_start = test_start
        self.acc_threshold = acc_threshold
        self.poi_emb_size = poi_emb_size

        self.epoch = epoch_max
        self.dropout_p = dropout_p
        self.use_cuda = True
        self.lr = lr
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.lr_schedule = lr_schedule
        self.optim = optim
        self.L2 = L2
        self.clip = clip
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.rnn_type = rnn_type
        self.app_emb_mode = app_emb_mode
        self.split_mode = split_mode
        self.model_mode = model_mode
        self.input_mode = input_mode
        self.attn_type = attn_type
		
        self.history_mode = history_mode
        self.baseline_mode = baseline_mode
