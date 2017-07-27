import numpy as np
import pandas as pd
import re

train_data = pd.read_json('train_data.json',orient="index")
test_data = pd.read_json('test_data.json',orient='index')

#set index
train_data.reset_index(inplace = True)
train_data.rename(columns={'index':'ID'}, inplace=True)

test_data.reset_index(inplace = True)
test_data.rename(columns={'index':'ID'}, inplace=True)


#check data
print ('Train data has {} rows and {} columns'.format(train_data.shape[0],train_data.shape[1]))
print ('test_data data has {} rows and {} columns'.format(test_data.shape[0],test_data.shape[1]))

#Encode Target Variable
train_data = train_data.replace({'segment':{'pos':1,'neg':0}})

#check target variable count
train_data['segment'].value_counts()/train_data.shape[0]

train_data['g1'] = [re.sub(pattern='\:\d+',repl='',string=x) for x in train_data['genres']]
train_data['g1'] = train_data['g1'].apply(lambda x: x.split(','))

train_data['g2'] = [re.sub(pattern='\:\d+', repl='', string = x) for x in train_data['dow']]
train_data['g2'] = train_data['g2'].apply(lambda x: x.split(','))


t1 = pd.Series(train_data['g1']).apply(frozenset).to_frame(name='t_genre')
t2 = pd.Series(train_data['g2']).apply(frozenset).to_frame(name='t_dow')

# using frozenset trick - might take few minutes to process
for t_genre in frozenset.union(*t1.t_genre):
    t1[t_genre] = t1.apply(lambda _: int(t_genre in _.t_genre), axis=1)

for t_dow in frozenset.union(*t2.t_dow):
    t2[t_dow] = t2.apply(lambda _: int(t_dow in _.t_dow), axis = 1)

train_data = pd.concat([train_data.reset_index(drop=True), t1], axis=1)
train_data = pd.concat([train_data.reset_index(drop=True), t2], axis=1)

test_data['g1'] = [re.sub(pattern='\:\d+',repl='',string=x) for x in test_data['genres']]
test_data['g1'] = test_data['g1'].apply(lambda x: x.split(','))

test_data['g2'] = [re.sub(pattern='\:\d+', repl='', string = x) for x in test_data['dow']]
test_data['g2'] = test_data['g2'].apply(lambda x: x.split(','))

t1_te = pd.Series(test_data['g1']).apply(frozenset).to_frame(name='t_genre')
t2_te = pd.Series(test_data['g2']).apply(frozenset).to_frame(name='t_dow')

for t_genre in frozenset.union(*t1_te.t_genre):
    t1_te[t_genre] = t1_te.apply(lambda _: int(t_genre in _.t_genre), axis=1)

for t_dow in frozenset.union(*t2_te.t_dow):
    t2_te[t_dow] = t2_te.apply(lambda _: int(t_dow in _.t_dow), axis = 1)

test_data = pd.concat([test_data.reset_index(drop=True), t1_te], axis=1)
test_data = pd.concat([test_data.reset_index(drop=True), t2_te], axis=1)

#the rows aren't list exactly. They are object, so we convert them to list and extract the watch time
w1 = train_data['titles']
w1 = w1.str.split(',')

#create a nested list of numbers
main = []
for i in np.arange(train_data.shape[0]):
    d1 = w1[i]
    nest = []
    nest = [re.sub(pattern = '.*\:', repl=' ', string= d1[k]) for k in list(np.arange(len(d1)))]
    main.append(nest)

blanks = []
for i in np.arange(len(main)):
    if '' in main[i]:
        # print "{} blanks found".format(len(blanks))
        blanks.append(i)
        
#replacing blanks with 0
for i in blanks:
    main[i] = [x.replace('','0') for x in main[i]]
    
#converting string to integers
main = [[int(y) for y in x] for x in main]

#adding the watch time
tosum = []
for i in np.arange(len(main)):
    s = sum(main[i])
    tosum.append(s)

train_data['title_sum'] = tosum

#making changes in test data
w1_te = test_data['titles']
w1_te = w1_te.str.split(',')

main_te = []
for i in np.arange(test_data.shape[0]):
    d1 = w1_te[i]
    nest = []
    nest = [re.sub(pattern = '.*\:', repl=' ', string= d1[k]) for k in list(np.arange(len(d1)))]
    main_te.append(nest)

blanks_te = []
for i in np.arange(len(main_te)):
    if '' in main_te[i]:
        # print "{} blanks found".format(len(blanks_te))
        blanks_te.append(i)
        
#replacing blanks with 0
for i in blanks_te:
    main_te[i] = [x.replace('','0') for x in main_te[i]]
    
#converting string to integers
main_te = [[int(y) for y in x] for x in main_te]

#adding the watch time
tosum_te = []
for i in np.arange(len(main_te)):
    s = sum(main_te[i])
    tosum_te.append(s)

test_data['title_sum'] = tosum_te

#count variables
def wcount(p):
    return p.count(',')+1

train_data['title_count'] = train_data['titles'].map(wcount)
train_data['genres_count'] = train_data['genres'].map(wcount)
train_data['cities_count'] = train_data['cities'].map(wcount)
train_data['tod_count'] = train_data['tod'].map(wcount)
train_data['dow_count'] = train_data['dow'].map(wcount)


test_data['title_count'] = test_data['titles'].map(wcount)
test_data['genres_count'] = test_data['genres'].map(wcount)
test_data['cities_count'] = test_data['cities'].map(wcount)
test_data['tod_count'] = test_data['tod'].map(wcount)
test_data['dow_count'] = test_data['dow'].map(wcount)


test_id = test_data['ID']
train_data.drop(['ID','cities','dow','genres','titles','tod','g1','g2','t_genre','t_dow'], inplace=True, axis=1)
test_data.drop(['ID','cities','dow','genres','titles','tod','g1','g2','t_genre','t_dow'], inplace=True, axis=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split


target = train_data['segment']
train_data.drop('segment',axis=1, inplace=True)


train_data["Sports"] = train_data.Boxing + train_data.Hockey + train_data.FormulaE + train_data.Cricket + train_data.Football + train_data.Tennis + train_data.Kabaddi + train_data.IndiaVsSa + train_data["Table Tennis"] + train_data.Volleyball + train_data.Athletics + train_data.Swimming + train_data.Formula1 + train_data.Badminton + train_data.Sport

test_data["Sports"] = test_data.Boxing + test_data.Hockey + test_data.FormulaE + test_data.Cricket + test_data.Football + test_data.Tennis + test_data.Kabaddi + test_data.IndiaVsSa + test_data["Table Tennis"] + test_data.Volleyball + test_data.Athletics + test_data.Swimming + test_data.Formula1 + test_data.Badminton + test_data.Sport

tt = train_data[["Travel","Reality","Romance","LiveTV","Comedy","Teen","NA","Horror","Awards","Science","Thriller",
               "Wildlife","Kids","TalkShow","Drama","Action","Mythology","Documentary","Family","Crime","Sports",
                "1","2","3","4","5","6","7","title_sum","title_count","genres_count","cities_count","tod_count",
                "dow_count"]]

tttest = test_data[["Travel","Reality","Romance","LiveTV","Comedy","Teen","NA","Horror","Awards","Science","Thriller",
               "Wildlife","Kids","TalkShow","Drama","Action","Mythology","Documentary","Family","Crime","Sports",
                "1","2","3","4","5","6","7","title_sum","title_count","genres_count","cities_count","tod_count",
                "dow_count"]]


import xgboost as xgb
gbm = xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.04).fit(tt, target)
pred_1 = gbm.predict_proba(tttest)


gbm = xgb.XGBClassifier(
    learning_rate = 0.05,
    n_estimators= 300,
    max_depth= 4,
    min_child_weight= 2,
    gamma=1,
    #gamma=0.9,                        
    subsample=0.8,
    colsample_bytree=0.8,
    #objective= 'binary:logistic',
    nthread= -1,
    scale_pos_weight=1).fit(tt, target)
pred_2 = gbm.predict_proba(tttest)

# Both the models above gave good results so I combined them giving higher weightage to second one since it performed # slightly well
pred = 0.45*pred_1+0.55*pred_2

#make submission file and submit
columns = ['segment']
sub = pd.DataFrame(data=pred[:,1], columns=columns)
sub['ID'] = test_id
sub = sub[['ID','segment']]
sub.to_csv("0.45xgb_0.55gbm.csv", index=False)

