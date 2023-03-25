import time
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import pickle
from tqdm import tqdm
tqdm.pandas()

# Determine whether or not GPU is available
print('Begin preprocessing')

nrow = None # How many rows we are reading from the original file

# skip = 2817700 # Skip the rows to start form user 942

# Load the data
path = "/home/jhnl/Usyd/cs48_usyd_capstone_2021"
usage = pd.read_csv('/'.join([path, 'data/App_usage_trace.txt']), delimiter=' ', nrows=nrow, names=['uid','datetime','loc','app_id','traffic'], dtype={'uid': int, 'datetime': str, 'loc': int, 'app_id': int, 'trafific': float})

usage['datetime'] = usage['datetime'].apply(lambda x: datetime.strptime(x, "%Y%m%d%H%M%S")) # Convert to datetime object
usage['day'] = usage['datetime'].apply(lambda x: x.day)

n_u = len(usage['uid'].unique())

n_users = 871
print("Successfully loaded data")
print("-"*30)
print('Retrieve {}/{} users'.format(n_u,n_users))

# Set the variable for interval span
span = 30
interval_span = str(span) + 'T'
n_intervals = int(60*24/span)
print('# of intervals:', n_intervals)

# Remove the weekend data to minimize memory usage
print('Remove weekend data')
usage = usage[(usage['datetime'] < datetime(2016, 4, 23)) | (usage['datetime'] > datetime(2016, 4, 25))]

print('Flooring the datetime to {} minutes interval'.format(span))
# Floor the datetime then convert it to h:m:s
#  according to the interval span such that we can aggregate the requests in a session
usage['floored_time'] = usage['datetime'].progress_apply(lambda x: pd.Timestamp.floor(x, freq=interval_span).time())

# Map the h:m:s to interval id e.g. 1, 2, 3, 4, 5
# Generate the intervals that will be matched to the timestamp of the data
intervals = pd.date_range('2020/1/1', freq=interval_span, periods=48)
intervals = [i.time() for i in intervals]

mapper = {}
for i, time in enumerate(intervals):
    mapper[time] = int(i)

# Map the id to the dataframe
usage['interval_id'] = usage['floored_time'].map(mapper)

# Drop the floored_time columns
usage = usage.drop(['floored_time', 'traffic', 'datetime'], axis=1)

print('Create multi-hot code App usage vector')
# Multihot code the app_id
app_np = usage['app_id'].to_numpy()

# Drop the app_id before doing numpy
usage = usage.drop('app_id', axis=1)

# Create the empty array to hold the one-hot-code app
app_multihot = np.zeros([len(usage), 2000]) # use int8 for less memory

# Set the corresponding app index to 1
for i, app in enumerate(app_np):
    app_multihot[i, app-1] = 1 # INDEXING STARTS FROM ZERO


# ADD POI ----------------------------------------------------------------------
base_poi = pd.read_csv('/'.join([path, 'data/base_poi.txt']), delimiter='\t')
poi_hot = base_poi.copy()
poi_hot= poi_hot.drop(columns=["BaseID"])
# change to one
for name in poi_hot.columns:
    poi_hot.loc[poi_hot[name] > 0, name] = 1
poi_hot["poi_hot"] = poi_hot.apply(lambda row: row.dropna().tolist(), axis=1)
poi_hot["loc"] = base_poi["BaseID"]
poi_hot = poi_hot[['loc','poi_hot']]
usage_multi = usage.merge(poi_hot, how='left', on='loc')
multi_hot_poi = np.stack(usage_multi['poi_hot'].to_numpy())
# --------------------------------------------------------------------------------

# print(usage.columns)
# Convert the dataframe into numpy array
usage_np = usage.to_numpy()
usage_np = np.concatenate([usage_np, app_multihot], axis=1) # add the uid, loc, day, session_id and app tgt
usage_np = np.concatenate([usage_np, multi_hot_poi], axis=1)

print('Add users to the final dataset')
# From here we start creating the data dictionary

data = {}
users = []
visited_session = []
session = np.zeros([2,2019])

with tqdm(total=n_u, desc='user handles', leave = False) as pbar:
    for n, i in enumerate(usage_np):

        u, loc, day, session_id, app, poi = int(i[0]), int(i[1]), i[2], i[3], i[4:2004], i[2004:]
        # try:
        if u == 942:
            continue
            
            
        if n == 0:

            # print('FIRST ROW')
            prev_session_id = session_id
            prev_u = u
            users.append(u)
            data[u] = {20:np.zeros([48,2019]), 21:np.zeros([48,2019]), 22:np.zeros([48,2019]), 25:np.zeros([48,2019]), 26:np.zeros([48,2019])}

        # Create a holding array for each user, padded with number of sessions per day
        if u not in users:

            # print('NOT IN USER')

            # Save the whole dictionary into a pickle file that could be run by the train.py
            file_name = '/'.join([path,'data/user_poi', '{}.pickle'.format(prev_u)])
            with open(file_name, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            prev_u = u
            
            data = {}
            session = np.zeros([2,2019])

            users.append(u)
            data[u] = {20:np.zeros([48,2019]), 21:np.zeros([48,2019]), 22:np.zeros([48,2019]), 25:np.zeros([48,2019]), 26:np.zeros([48,2019])}
            
            pbar.update(1) # For tracking the pre-processing
        
        # If the next request is still in the same interval
        if session_id == prev_session_id:
            request = np.expand_dims(np.concatenate([[session_id], [loc], app, poi]), axis=0)
            session = np.concatenate([session,request])
            # print("pot", poi)
            # print("session_poi", session[:,2002:])
            # print(session.shape)
            # prev_session_id = session_id

        elif session_id != prev_session_id:
            # print('NEW INTERVAL')
            # Create a session
            s_loc = np.expand_dims(stats.mode(session[:, 1]).mode, axis=0)#.astype(np.int16)
            s_app = np.expand_dims(np.sum(session[:, 2:2002], axis=0), axis=0)#.astype(np.int16)
            # print(stats.mode(session[:, 1]).mode[0])
            s_poi = np.expand_dims(poi_hot.loc[poi_hot['loc'] == stats.mode(session[:, 1]).mode[0], 'poi_hot'].iloc[0], axis=0)
            # print(s_poi)
            # s_poi = np.expand_dims(np.sum(session[:,2002:], axis=0), axis=0)
            s_sid = np.expand_dims([prev_session_id], axis=0)#.astype(np.int16)

            # print('SET THE INTERVAL')
            # Set the interval of the day
            data[u][day][int(prev_session_id)] = np.concatenate([s_sid, s_loc, s_app, s_poi], axis=1) 

            # print('RESET SESSION')
            # Reset the session initiated by the new request in the sessino
            session = np.expand_dims(np.concatenate([[session_id], [loc], app, poi]), axis=0)

        prev_session_id = session_id

        # except:
        #     print(n, u, loc, day, session_id)
        #     break




print('Done with all users!')




