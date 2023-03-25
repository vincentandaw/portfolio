# Change path at line 39, 183 and 235 to local corresponding path
import numpy as np
import pandas as pd
from scipy import stats
from collections import deque, Counter
from sklearn import metrics as skmetrics
import torch
from torch.autograd import Variable

# Determine whether or not GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('Using {} device'.format(device))

# Read in the weather data for generate_input()
weather = pd.read_csv('./DeepAppmodel/data/weather_data.txt', delimiter='\t')
weather_np= weather.T.to_numpy()
weather_dict = {20:weather_np[0],21:weather_np[1],22:weather_np[2],23:weather_np[2],24:weather_np[2],25:weather_np[3],26:weather_np[4]} #Load the weather data

# Helper function that formats the data to generate the input that will be imputed in the model
# Two modes are available 'temporal' and 'user'
# If temporal is chosen 20,21,22 are the training set, 25 the validation set, 26 the training set
# If user is chosen all days, including the week end are included and the 3 sets are splited based on the split_ratio parameter [trainset ratio, val-set ratio,testset ratio]
def generate_input(data, mode, split_mode,seed=42,split_ratio=[0.8,0.1,0.1], n_intervals=48, user_topk=None, mode2=None, candidate=None):
    data_train = {}
    train_idx = {}
    data_neural = data


    if split_mode == 'temporal':
        if mode == 'train':
            train_id = [20, 21, 22] #the day for training, add 23 in the list if week end data is included
        elif mode == 'val':
            train_id = [25] # the day for validation
        elif mode == 'test':
            train_id = [26] #add 24 in the list if week end data is included
    elif split_mode == 'user':
        train_id = [20,21,22,23,24,25,26] #If user split, keep all 7 days data
    else:
        raise ValueError('Invalid split_mode')

    if candidate is None:
        candidate = list(data_neural.keys()) #do not filter 

    if split_mode == 'user': # If user mode chosen
        ratio_check = 0
        for i in split_ratio: #Check if the given ratio is valid
            ratio_check+=i 
        if ratio_check != 1:
            raise ValueError('Invalid split_ratio')
        np.random.seed(seed) #Set random seed  to get the same split
        np.random.shuffle(candidate) #shuffle the users id
        train, val, test = \
            np.split(candidate, 
                    [int(split_ratio[0]*len(candidate)), int((split_ratio[0]+split_ratio[1])*len(candidate))]) #split the data into corresponding partitions
        # print(test)
        if mode == 'train':
            candidate = train #if train mode, keep train split users as candidate
        elif mode == 'val':
            candidate = val #if val mode, keep train split users as candidate
        elif mode == 'test':
            candidate = test #if test mode, keep train split users as candidate
        
    for u in candidate:
        # sessions = data_neural[u] # Get all the sessions for all days

        
        data_train[u] = {} #re-index user from 0

        for c, i in enumerate(train_id): # 20,21,22
            session = data[u][i]
            trace = {}
            tim_np = np.reshape(np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1))
            ptim_np = np.reshape(np.array([s[0] for s in session[1:]]), (len(session[:-1]), 1))
            loc_np = np.reshape(np.array([s[1] for s in session[:-1]]), (len(session[:-1]), 1))
            app_np = np.reshape(np.array([s[2:2002] for s in session[:-1]]), (len(session[:-1]), 2000))
            loc_target = np.array([s[1] for s in session[1:]])
            loc_loss =  np.array([s[1] for s in session[:-1]])
            poi_np = np.reshape(np.array([s[2002:] for s in session[:-1]]), (len(session[:-1]), 17))
            # poi_onehot = torch.LongTensor(len(poi_np),17)
            app_target = np.reshape(np.array([s[2:2002] for s in session[1:]]), (len(session[:-1]), 2000))
            app_target = np.where(app_target > 0, 1, 0) # Multi-hot code the output vector with 1, 0
            uid_target = np.array([int(u)]).repeat(len(tim_np), 0)
            weather = weather_dict[i][:-1]
            
            tim_onehot = torch.FloatTensor(len(tim_np),n_intervals)
            tim_onehot.zero_()
            tim_onehot.scatter_(1,torch.LongTensor(tim_np),1)
            
            ptim_onehot = torch.FloatTensor(len(ptim_np),n_intervals)
            ptim_onehot.zero_()
            ptim_onehot.scatter_(1,torch.LongTensor(ptim_np),1)
            
            loc_onehot = torch.FloatTensor(len(loc_np),10000)
            loc_onehot.zero_()
            loc_onehot.scatter_(1,torch.LongTensor(loc_np),1)

            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            trace['ptim'] = Variable(torch.LongTensor(ptim_np))
            trace['loc_o'] = Variable(loc_onehot)
            trace['tim_o'] = Variable(tim_onehot)
            trace['ptim_o'] = Variable(ptim_onehot)
            trace['app'] = Variable(torch.FloatTensor(app_np))
            trace['loc_target'] = Variable(torch.LongTensor(loc_target))
            trace['loc_loss'] = Variable(torch.LongTensor(loc_loss))
            trace['app_target'] = Variable(torch.FloatTensor(app_target))
            trace['uid_target'] = Variable(torch.LongTensor(uid_target))
            trace['poi'] = Variable(torch.FloatTensor(poi_np))
            trace['weather'] = Variable(torch.FloatTensor(weather))

            data_train[u][i] = trace
        train_idx[u] = train_id
        if user_topk is not None:
            data_train[u]['loc_topk'] = Variable(torch.LongTensor(user_topk[u][0]))
            data_train[u]['app_topk'] = Variable(torch.FloatTensor(user_topk[u][1]))

    print('The number of users: {0} '.format(len(candidate)))
    return data_train, train_idx # Engineered features, day

#Helper function that generates a queue of user data that will be imputed to the model during the training or testing process - Is used in run_simple method
def generate_queue(train_idx, mode, mode2, inputmode='short'):
	"""return a deque. You must use it by train_queue.popleft()"""
	user = list(train_idx.keys())
	train_queue = deque()
	if mode == 'random':
		initial_queue = {}
		for u in user:
			if mode2 == 'train':
				if inputmode == 'long' or inputmode == 'short_history':
					initial_queue[u] = deque(train_idx[u])#[1:]) 
				elif inputmode == 'short':
					initial_queue[u] = deque(train_idx[u])
			else:
				initial_queue[u] = deque(train_idx[u])
		queue_left = 1
		while queue_left > 0:
			np.random.shuffle(user)
			for j, u in enumerate(user):
				if len(initial_queue[u]) > 0:
					train_queue.append((u, initial_queue[u].popleft()))
				if j >= int(0.01 * len(user)):
					break
			queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
	elif mode == 'normal':
		for u in user:
			for i in train_idx[u]:
				train_queue.append((u, i))
	return train_queue

#Helper function that runs the model for either testing or training depending on 'mode' parameter 
def run_simple(data, run_idx, mode, inputmode, lr, clip, model, optimizer, criterion, mode2=None,test_start=2, threshold=0.3):
    """mode=train: return model, avg_loss
        mode=test: return avg_loss,avg_acc,users_rnn_acc"""
    run_queue = None
    model.to(device)

    if mode == 'train':
        model.train(True)
        run_queue = generate_queue(run_idx, 'random', 'train',inputmode)
        print('Run_queue for training:', len(run_queue)) #(u,i)list
    elif mode == 'test':
        model.train(False)
        run_queue = generate_queue(run_idx, 'normal', 'test',inputmode)
        print('Run_queue for testing:',len(run_queue))
    elif mode == 'train_test': #test the training data set 
        model.train(False)
        run_queue = generate_queue(run_idx, 'random', 'train',inputmode)
        print('Run_queue for training:', len(run_queue)) #(u,i)list

    total_loss = []
    total_loss_app = []
    total_loss_loc = []
    total_loss_uid = []
    queue_len = len(run_queue)

    save_prediction = {}

    users_acc = {}
    for c in range(queue_len): # refer this to a batch in mini-batch training
        optimizer.zero_grad()
        u, i = run_queue.popleft() #u is the user, i is the train_ID
        if u not in save_prediction:
            save_prediction[u] = {}
        if u not in users_acc:
            users_acc[u] = {'tim_len':0, 'app_acc':[0,0,0,0,0], 'loc_acc':[0,0,0], 'uid_acc':[0,0], 'n_interval': 1} 
        loc = data[u][i]['loc'].to(device)
        loc_loss = data[u][i]['loc_loss'].to(device)
        tim = data[u][i]['tim'].to(device)
        ptim = data[u][i]['ptim'].to(device)
        app = data[u][i]['app'].to(device)
        app_loss = data[u][i]['app'].to(device)
        app_target = data[u][i]['app_target'].to(device)
        loc_target = data[u][i]['loc_target'].to(device)
        uid_target = data[u][i]['uid_target'].to(device)
        uid = Variable(torch.LongTensor([int(u)])).to(device)
        poi = data[u][i]['poi'].to(device)
        rain = data[u][i]['weather'].to(device)
        
        #save_prediction[u][i] = {}
        #save_prediction[u][i]['tim'] = tim.data.cpu().numpy()
            
        """model input"""
        if mode2 in ['AppPreLocPreUserGtrTopk']:	
            app_topk = data[u]['app_topk'].to(device)
            loc_topk = data[u]['loc_topk'].to(device)
            app_scores,loc_scores = model(tim, app, loc, uid, ptim,app_topk,loc_topk).to(device)

        elif 'AppPreLocPreUserIdenGtrLinear' in mode2:
            tim_o = data[u][i]['tim_o'].to(device)
            ptim_o = data[u][i]['ptim_o'].to(device)
            loc_o = data[u][i]['loc_o'].to(device)
            app_scores, loc_scores, uid_scores = model(tim_o, app, loc_o, uid, ptim_o)		   
        elif mode2 in ['AppPreLocPreUserIdenGtr']:
            app_scores, loc_scores, uid_scores = model(tim, app, loc, uid, ptim)   
        elif mode2 in ['AppPreLocPreUserIdenPOIGtr', 'AppPreLocPreUserIdenPOIFCGtr']:
            app_scores, loc_scores, uid_scores = model(tim, app, loc, uid, ptim, poi)
        elif mode2 in ['AppPreLocPreUserIdenPOIWeatherGtr','AppPreLocPreUserIdenPOIWeatherFCGtr']:
            app_scores, loc_scores, uid_scores = model(tim, app, loc, uid, ptim, poi, rain)
        
        # """RNN output cut"""
        if 'AppPre' in mode2:
            app_scores = app_scores[-(app_target.data.size()[0]-test_start):] #Skip the first n predictions??
            app_target = app_target[-app_scores.data.size()[0]:] # same size as app_score
            app_loss = app_loss[-app_scores.data.size()[0]:] # ensure same size
            targrt_len = len(app_scores)
        # if 'LocPre' in mode2:
        #     loc_scores = loc_scores[-(loc_target.data.size()[0]-test_start):]
        #     loc_target = loc_target[-loc_scores.data.size()[0]:]
        #     loc_loss = loc_loss[-loc_scores.data.size()[0]:]
        #     targrt_len = len(loc_scores)
        # if 'UserIden' in mode2:
        #     uid_scores = uid_scores[-(uid_target.data.size()[0]-test_start):]
        #     uid_target = uid_target[-uid_scores.data.size()[0]:]
        #     targrt_len = len(uid_scores)

        """model loss"""	
        if mode2 in ['AppPreLocPreUserIdenGtr','AppPreLocPreUserIdenPOIGtr','AppPreLocPreUserIdenGtrLinear', 'AppPreLocPreUserIdenPOIFCGtr','AppPreLocPreUserIdenPOIWeatherGtr','AppPreLocPreUserIdenPOIWeatherFCGtr']:
            loss, loss_app, loss_loc, loss_uid = criterion(app_scores, app_target, loc_scores, loc_target, uid_scores, uid_target)
            total_loss_app.append(loss_app.data.cpu().numpy())
            total_loss_loc.append(loss_loc.data.cpu().numpy())
            total_loss_uid.append(loss_uid.data.cpu().numpy())
        else:
            print("The loss function is not stated correctly.")
            
            
        if mode == 'train':
            loss.backward() 
            try: # gradient clipping
                torch.nn.utils.clip_grad_norm(model.parameters(), clip) #clip = 1, 1 norm
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.add_(p.grad.data, alpha=-lr)
            except:
                pass
            optimizer.step()

        elif mode == 'test':

            # Hard code the number of interval in a window (24 hours)
            users_acc[u]['tim_len'] += 48

            # In each popped item (data of 1 user in 1 day), we calculate the AUC, 
            # average precision, precision and recall and save it in users_acc
            if 'AppPre' in mode2:
                # print('APP_TARGET',app_target)
                # We ignore the predictions when the user does not have any record in that day
                if torch.sum(app_target) <= 0:
                    pass
                else:
                    app_acc, n_record = get_acc(app_target, app_scores, threshold)
                    users_acc[u]['app_acc'][0] += app_acc[0] #AUC
                    users_acc[u]['app_acc'][1] += app_acc[1] #F1
                    users_acc[u]['app_acc'][2] += app_acc[2] #P
                    users_acc[u]['app_acc'][3] += app_acc[3] #R
                    users_acc[u]['app_acc'][4] += app_acc[4] #f1
                    users_acc[u]['n_interval'] += n_record
            # if 'LocPre' in mode2:
                # loc_acc = get_acc2(loc_target, loc_scores)
                # users_acc[u]['loc_acc'][0] += loc_acc[0] #top1
                # users_acc[u]['loc_acc'][1] += loc_acc[1] #top5
                # users_acc[u]['loc_acc'][2] += loc_acc[2] #top10
            # if 'UserIden' in mode2:
                # uid_acc = get_acc2(uid_target, uid_scores)
                # users_acc[u]['uid_acc'][0] += uid_acc[0] #top1
                # users_acc[u]['uid_acc'][1] += uid_acc[2] #top10

            
                
        total_loss.append(loss.data.cpu().numpy())

    avg_loss = np.mean(total_loss)
    # if len(total_loss_app)>0:
    #     print('{} App Loss:{:.4f}'.format(mode, np.mean(total_loss_app)))
    # if len(total_loss_loc)>0:
    #     print('{} Loc Loss:{:.4f}'.format(mode, np.mean(total_loss_loc)))
    # if len(total_loss_uid)>0:
    #     print('{} Uid Loss:{:.4f}'.format(mode, np.mean(total_loss_uid)))


    if mode == 'train':
        return model, avg_loss, save_prediction

    elif mode == 'test':
        user_acc = {}  #average acc for each user

        for u in users_acc:
            user_acc[u] = {'app_auc':0, 'app_map':0, 'app_precision':0, 'app_recall':0, 'loc_top1':0, 'loc_top5':0, 'loc_top10':0} 
            user_acc[u]['app_auc'] = users_acc[u]['app_acc'][0] / users_acc[u]['n_interval'] #users_acc[u]['tim_len']
            user_acc[u]['app_map'] = users_acc[u]['app_acc'][1] / users_acc[u]['n_interval'] #users_acc[u]['tim_len']
            user_acc[u]['app_precision'] = users_acc[u]['app_acc'][2] / users_acc[u]['n_interval'] #users_acc[u]['tim_len']
            user_acc[u]['app_recall'] = users_acc[u]['app_acc'][3] / users_acc[u]['n_interval'] #users_acc[u]['tim_len']
            user_acc[u]['app_f1'] = users_acc[u]['app_acc'][4] / users_acc[u]['n_interval']
            # user_acc[u]['loc_top1'] = users_acc[u]['loc_acc'][0] / users_acc[u]['tim_len']
            # user_acc[u]['loc_top5'] = users_acc[u]['loc_acc'][1] / users_acc[u]['tim_len']
            # user_acc[u]['loc_top10'] = users_acc[u]['loc_acc'][2] / users_acc[u]['tim_len']
            # user_acc[u]['uid_top1'] = users_acc[u]['uid_acc'][0] / users_acc[u]['tim_len']
            # user_acc[u]['uid_top10'] = users_acc[u]['uid_acc'][1] / users_acc[u]['tim_len']
        avg_acc = {}
        avg_acc['app_auc'] = np.mean([user_acc[u]['app_auc'] for u in users_acc]) 
        avg_acc['app_map'] = np.mean([user_acc[u]['app_map'] for u in users_acc]) 
        avg_acc['app_precision'] = np.mean([user_acc[u]['app_precision'] for u in users_acc]) 
        avg_acc['app_recall'] = np.mean([user_acc[u]['app_recall'] for u in users_acc])
        avg_acc['app_f1'] = np.mean([user_acc[u]['app_f1'] for u in users_acc])
        # avg_acc['loc_top1'] = np.mean([user_acc[u]['loc_top1'] for u in users_acc])
        # avg_acc['loc_top5'] = np.mean([user_acc[u]['loc_top5'] for u in users_acc])
        # avg_acc['loc_top10'] = np.mean([user_acc[u]['loc_top10'] for u in users_acc])
        # avg_acc['uid_top1'] = np.mean([user_acc[u]['uid_top1'] for u in users_acc])
        # avg_acc['uid_top10'] = np.mean([user_acc[u]['uid_top10'] for u in users_acc])
        return avg_loss, avg_acc, user_acc, save_prediction

    elif mode == 'train_test':
        return model, avg_loss, save_prediction

#Helper function that helps calculates the evaluation metrics
def get_acc(target, scores, threshold): #AUC and F1 for Binary classification
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    scores = scores.data.cpu().numpy()
    acc = np.zeros(5) # AUC, AP, P, R, f1
    record = 0 # Count the number of interval with records for averaging 
    for i in range(len(target)):
        truth = target[i,:]
        # print(truth)
        if np.sum(truth) > 0:
            predict = scores[i,:]
            # predict_b = cal_preb(predict,5)
            # This let us always focus on the top5 prediction with a threshold 0.5 in the sigmoid
            actual, pred_o, pred = r_k(truth, predict, 5, 0.5)

            fpr, tpr, thresholds = skmetrics.roc_curve(truth, predict, pos_label=1) # Collect the recall and false positive rate from all 2000 predictions
            acc[0] += skmetrics.auc(fpr, tpr)
            acc[1] += cal_ap(truth, predict, 5) # mean average precision (MAP) over 5 apps
            acc[2] += skmetrics.precision_score(actual, pred_o, average='macro') # Precision@5
            acc[3] += skmetrics.recall_score(actual, pred_o, average='macro') # Recall@5
            acc[4] += skmetrics.f1_score(actual, pred_o, average='macro') # f1

            # acc[5] += skmetrics.precision_score(truth, predict, average='macro') # Precision
            # acc[5] += skmetrics.recall_score(truth, predict, average='macro') # Recall
            # acc[7] += skmetrics.f1_score(truth, predict, average='macro') # f1
            record +=1
        else:
            pass
    
    return acc, record

def get_acc2(target, scores): #TOPK acc for loc/user classification
	"""target and scores are torch cuda Variable"""
	target = target.data.cpu().numpy()
	val, idxx = scores.data.topk(10, 1) #torch.topk to get top elements at dim=1
	predx = idxx.cpu().numpy()
	acc = np.zeros(3)
	for i, p in enumerate(predx):
		t = target[i]
		if t in p[:10]:
			acc[2] += 1
		if t in p[:5]:
			acc[1] += 1
		if t == p[0]:
			acc[0] += 1
	return acc

# They use this to calculate the mean average precision (MAP) cal_ap means calculate average precision
def cal_ap( y_actual, y_pred, k ):
    topK = min( len(y_pred), k ) # set top k
    l_zip = list(zip(y_actual,y_pred))
    # sort y_pred by the probability of the model
    s_zip = sorted( l_zip, key=lambda x: x[1], reverse=True )
    # topk of sorted result
    s_zip_topk = s_zip[:topK] # Shape (5,2)

    # Calculation of precision
    num = 0
    rank = 0
    sumP = 0.0
    for item in s_zip_topk:
        rank += 1
        if item[0] == 1:
            num += 1
            sumP += (num*1.0)/(rank*1.0)
    ap = 0.0
    if num > 0:
        ap = sumP/(num*1.0)
    return ap	# average precision

def cal_preb(y_pred, k):
    topK = min( len(y_pred), k )
    # Create a list of (idx, probability) then sort by probability
    l_zip = list(zip(range(len(y_pred)),y_pred))
    s_zip = sorted( l_zip, key=lambda x: x[1], reverse=True )
    # topk of highest probability
    pre_b = np.zeros(len(y_pred)) #pre_b shape (2000,)
    s_zip_topk = s_zip[:topK]
    for (index,sore) in s_zip_topk:
        pre_b[index] = 1
    return pre_b # return one hot-code of but only the topk apps


# Take topk prediction and the ground truth
def r_k(y_actual, y_pred, k, threshold):
    topK = min( len(y_pred), k ) # set top k
    l_zip = list(zip(y_actual,y_pred))
    # sort y_pred by the probability of the model
    s_zip = sorted( l_zip, key=lambda x: x[1], reverse=True )
    # topk of sorted result
    s_zip_topk = s_zip[:topK] # Shape (5,2)

    actual, pred = zip(*s_zip_topk)
    actual = np.where(np.array(actual) > threshold, 1, 0)

    pred = np.array(pred)
    pred_o = np.where(pred > threshold, 1, 0)

    return actual, pred_o, pred