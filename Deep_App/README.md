# DeepApp Replicate

Here we reverse engineer the DeepApp Architecture. Since that the DeepApp was written in Python 2.7 and and older version of Pytorch, the code is now re-written using Python 3.9 and Pytorch 1.8.1. Besides, from the code base, we also removed the functions that are not required for DeepApp. There are 5 python file here.


## Instruction
1. Download the following folders
    - Normal data - 
    - Removed Oscillation data - 
2. Move the data folders to ```./DeepAppmodel/data```

### train.py
1. Choose and set the correct model type such as 'AppPreLocPreUserIdenPOIFCGtr'
2. Run the train.py
### test.py
1. Choose and set the correct model type such as 'AppPreLocPreUserIdenPOIFCGtr'
2. Choose the set correct model file such as 'AppPreLocPreUserIdenPOIFCGtr2021-05-21 02:06:23'
3. Run the test.py 

## Current issue observed
1. Source code did not to train val test split. Instead, they train on first 3 days then predict last two days
2. Source code use app count as target
3. Data fed to the network is not in chronological order (user by user)
4. Can we frame the question as classification? Or is RNN better?
5. Is there an RNN with short memory only?


## Evalutation metrics

### Recall@K from AppUsage2Vec
Take the topk Apps as prediction and calculate the recall
```
acc[3] += skmetrics.recall_score(actual, pred_o, average='macro')
```
actual and pred_0 are one-hot code with length == 5


### Mean Average Precision (MAP) over 2000 apps
Different from precision!!! Ref: https://towardsdatascience.com/map-mean-average-precision-might-confuse-you-5956f1bfa9e2 <br> DeepApp used a customized function ```cal_ap()``` in ```helper.py``` 

### AUC of all 2000 apps
```
# Collect the recall and false positive rate from all 2000 predictions
fpr, tpr, thresholds = skmetrics.roc_curve(truth, predict, pos_label=1) 

# Calculate the auc
acc[0] += skmetrics.auc(fpr, tpr)
```

## Current issue
1. (SOLVED) App accuracy is abnormally high which is likely due to zero-padding so that there are many unwanted correct prediction (Only consider the interval with at least 1 record, ignore all empty predictions)
2. (SOLVED) Memory issue to preprocess user 942 with over 1M records (Preprocess User 942 separately)
3. (SOLVED) Loss function for uid is not working properly (torch.NLLLoss)
4. (SOLVED) They use a special way to calculate the f1-score in the get_acc() in ```helper.py``` (Modified the code according to paper)


## Different python file
### preprocess / preprocess_peru.py
From the DeepApp paper, they preprocess the data according to their definition. 
They use 30-min interval as a session. A window is 24 hours which consists of 48 intervals. 
They did not implement a dataloader. Rather they treat each window as a batch. (**Can be fine-tuned**)

```preprocess.py``` runs all user at once and will likely run out of memory. ```preprocess_peru.py``` preprocess each user.


The preprocessed data should look like this: 
```
data: {
    user: 
        '20-Apr': {
            'tim' : [list of time in the window Shape(48,1)],
            'loc' : [lsit of loc in the window Shape(48,1)],
            'app': [list of multi-hot-code vector in the window. Shape(48, 2000)]    
        },
        '21-Apr': {
            'tim' : [list of time in the window Shape(48,1)],
            'loc' : [lsit of loc in the window Shape(48,1)],
            'app': [list of multi-hot-code vector in the window. Shape(48, 2000)]
        }, ........
    },
    user2: {
        Same pattern........
    }
}
```

### helper.py
- generate_input returns a train set and a test set, adds more field like ptim, app_target, loc_target, uid, tim_o, loc_o, topk, etc.
- generate_queue returns a queue so it pops a window of a user everytime and feed to the model

In other words, if we use a dataloader instead. For each user, trainloader has 3 batches, Vall has 1. Each batch has 48 samples. If we want to use dataloader instead, we have to make sure each iteration of it contains the information of one user?? Or order does not matter at all.


### model.py
Contains the model design

### train.py
This file loads in all the preprocessed user data into a single dictionary. Then it initiate a ```RnnParameterData``` object that contains all the parameters required for the model. Finally, it run the model training, testing, etc. The code for visualiztion is commented out for now.

### Possible model type

AppPreLocPreUserIdenGtr - Original DeepApp model
AppPreLocPreUserIdenPOIGtr - DeepApp with additional POI data
AppPreLocPreUserIdenPOIFCGtr - DeepApp with additional POI data with fully connected layer instead of GRU
AppPreLocPreUserIdenPOIWeatherGtr - DeepApp with additional POI and Weather data
AppPreLocPreUserIdenPOIWeatherFCGtr - DeepApp with additional POI and weather data with fully connected layer instead of GRU

