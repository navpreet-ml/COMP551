"""
Please run the whole code directly. It will show the results we reported in table. 2 directly.
"""
import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import pandas as pd
import numpy as np
import collections as cl
import time
with open("proj1_data.json") as fp:
    data_wP = json.load(fp)
    data_noP = data_wP

#Convert JSON to PANDAS dataframe
data_wP = pd.DataFrame.from_dict(data_wP, orient = 'columns')
data_noP = pd.DataFrame.from_dict(data_noP, orient = 'columns')

### With punctuation x [7 - 167] ====================================================================
dataMF_wP = data_wP[:10000]
#Lower case, spliting at white space and 160 most frequent words with the frequency
mostFreq_wP = np.array(cl.Counter(" ".join(dataMF_wP["text"].str.lower()).split()).most_common(160))
#Just the most frequenct words, no frequency
mostFreq_wP = mostFreq_wP[:,0]
#Pre-processing the text
#Forming vector x(160x1) for each comment with elements 1 and 0, with 1 corresponding to a match between the words of mostFreq and the comment 
data_wP["text"] = data_wP["text"].str.lower().str.split(" ")
#Declare zeros of 12000x60
x_wP = np.zeros((12000,160))    
for i in range(0,data_wP.shape[:][0]):
    for word in data_wP["text"][i]:
        for j in range(0, mostFreq_wP.shape[:][0]):
            if word == mostFreq_wP[j]:
                x_wP[i][j] = x_wP[i][j] + 1

### Without punctuation x [7 - 167] ====================================================================
#Remove punctuation
data_noP["text"] = data_noP['text'].str.replace('[^\w\s]','')
#take only first 10000 data points
dataMF_noP = data_noP[:10000]
#Lower case, spliting at white space and 160 most frequent words with the frequency
mostFreq_noP = np.array(cl.Counter(" ".join(dataMF_noP["text"].str.lower()).split()).most_common(160))
#Just the most frequenct words, no frequency
mostFreq_noP = mostFreq_noP[:,0]
#Pre-processing the text
#Forming vector x(160x1) for each comment with elements 1 and 0, with 1 corresponding to a match between the words of mostFreq and the comment 
data_noP["text"] = data_noP["text"].str.lower().str.split(" ")
#Declare zeros of 12000x60
x_noP = np.zeros((12000,160))    
for i in range(0,data_noP.shape[:][0]):
    for word in data_noP["text"][i]:
        for j in range(0, mostFreq_noP.shape[:][0]):
            if word == mostFreq_noP[j]:
                x_noP[i][j] = x_noP[i][j] + 1

### Create x [0 (Bias), 1 - 4 (3 features)] & y ====================================================================
data = data_wP
#Converting true and false values inot binaries
data["is_root"] = data["is_root"].astype(int)
#Store Popularity as target variable
y = data["popularity_score"]
y = y.values.reshape(12000,1)
#Dropping target variable in data
data = data.drop(['popularity_score','text'],1)

### Create New features x [4 - 7] ====================================================================
data['child_sq'] = data['children']**2 # square
data['log_term'] = np.log(data['children'] + 0.0000001) # log
data['play'] = data['children']**3 # play

### total x [1 - 167] ====================================================================
data_wP = data
data_noP = data
#Add 60 new vectors from the most frequent
data_wP = np.concatenate((data_wP, x_wP),axis = 1)
data_noP = np.concatenate((data_noP, x_noP),axis = 1)
#declare bias matrix
singles_wP = np.ones((12000,1))
singles_noP = np.ones((12000,1))
#Add bias matrix
X_wP = np.concatenate((singles_wP, data_wP),axis=1)
X_noP = np.concatenate((singles_noP, data_noP),axis=1)

def closed_form (X_train, y_train): 
#    start = time.time()
    #Closed loop implementation
    xTx = X_train.T.dot(X_train)
    xTy = X_train.T.dot(y_train)
    xTxINV = np.linalg.inv(xTx)
    w = xTxINV.dot(xTy)
#    end = time.time()
#    print("Runtime: ", end - start)
    return w

def gradient_descent(x, y, n, beta, epsilon):
    np.random.seed(123) #Set the seed
    w = pd.DataFrame(np.random.rand(x.shape[1])) #choose random starting value
    grad = epsilon+1 #initializes while loop, will always be greater than epsilon\n",
#    MSE_history = pd.DataFrame(MSE(x,y,w))
    p1 = np.dot(x.T,x)
    p2 = np.dot(x.T, y)
    alpha = n/(1+beta)
    while (abs(grad) > epsilon):
        if (alpha>1e-5):
            alpha = n/(1+beta)
            beta = 10*beta #want beta to get bigger with each iteration so that the learning rate decays\n",
        gradient = 2*(np.dot(p1,w)-p2)
        grad = np.linalg.norm(gradient)    
        w = w-alpha*(gradient)
 #       MSE_history = MSE_history.append(pd.DataFrame(MSE(x,y,w)), ignore_index = True)
    return w #,  MSE_history"

def MSE(x,y,w):
    MSE = np.sum(pd.DataFrame((np.square(np.dot(x,w)-y))/len(y)))
    return MSE

#Splitting Data
def TRA(x):
    return x[:10000]
def VAL(x):
    return x[10000:11000]
def TEST(x):
    return x[11000:12000]

# x selection [Backward] ===============================================================
def x_select_b(X, y):
    X_train_best_back = pd.DataFrame(X[:10000]) #start with all features
    X_val_best_back = pd.DataFrame(X[10000:11000])
    X_test_best_back = pd.DataFrame(X[11000:12000])
    Y_train = y[:10000]
    Y_val = y[10000:11000]
    w_new_back = closed_form(X_train_best_back, Y_train) #find the weights 
    mse_val_previous_back = pd.DataFrame(MSE(X_val_best_back, Y_val, w_new_back)) #find mse
    for i in range(0,X.shape[1]):
        X_train_check_back = X_train_best_back.drop([i], axis=1)
        X_val_check_back = X_val_best_back.drop([i], axis=1)
        w_new_back = closed_form(X_train_check_back, Y_train)
        mse_val_back = pd.DataFrame(MSE(X_val_check_back, Y_val, w_new_back))
        if(mse_val_back[0][0]<mse_val_previous_back[0][0]):
            X_train_best_back = X_train_best_back.drop([i], axis=1)
            X_val_best_back = X_val_best_back.drop([i], axis=1)
            X_test_best_back = X_test_best_back.drop([i], axis=1)
        mse_val_previous_back = mse_val_back
    return pd.concat([X_train_best_back, X_val_best_back, X_test_best_back], axis = 0)

## x selection [Forward] =============================================================== 
def x_select_f(X, y):
    X_train_best = pd.DataFrame(X[:10000]).iloc[:,0:1] #start with feature number 0
    X_val_best = pd.DataFrame(X[10000:11000]).iloc[:,0:1]
    X_test_best = pd.DataFrame(X[11000:12000]).iloc[:,0:1]
    Y_train = y[:10000]
    Y_val = y[10000:11000]
    w_new = closed_form(X_train_best, Y_train) #find the weights that just that feature gives
    mse_val_previous = pd.DataFrame(MSE(X_val_best, Y_val, w_new)) #find mse
    for i in range(2,X.shape[1]+1):
        X_train_new = pd.DataFrame(X[:10000]).iloc[:,i-1:i] #gives feature i-1
        X_val_new =  pd.DataFrame(X[10000:11000]).iloc[:,i-1:i]
        X_test_new = pd.DataFrame(X[11000:12000]).iloc[:,i-1:i]
        X_train_check = pd.concat([X_train_best, X_train_new], axis = 1)
        X_val_check = pd.concat([X_val_best, X_val_new], axis = 1)
        w_new = closed_form(X_train_check, Y_train)
        mse_val = pd.DataFrame(MSE(X_val_check, Y_val, w_new))
        if(mse_val[0][0]<mse_val_previous[0][0]):
            X_train_best = pd.concat([X_train_best, X_train_new], axis =1)
            X_val_best = pd.concat([X_val_best, X_val_new], axis = 1)
            X_test_best = pd.concat([X_test_best, X_test_new], axis = 1)
        mse_val_previous = mse_val
    return pd.concat([X_train_best, X_val_best, X_test_best], axis = 0)

X = X_wP[...,:4]
print ("01 Closed-form: 3f: ")
w = closed_form(TRA(X), TRA(y)) # Only considering 3 features
print ("Train err: ", MSE(TRA(X), TRA(y), w).values, "Valid err: ", MSE(VAL(X), VAL(y), w).values)
print ("We start with this as a reference")
print ("\n")
X = X_wP[...,:4] # Only considering 3 features
print ("01 Gradient descent: 3f: ")
w = gradient_descent(TRA(X), TRA(y), 0.0001, 0.00001, 0.000001)
print ("Train err: ", MSE(TRA(X), TRA(y), w).values, "Valid err: ", MSE(VAL(X), VAL(y), w).values)
print ("We start with this as a reference")
print ("\n")
#=================================================================
X = np.concatenate((X_wP[...,:4], X_wP[...,7:67]),axis=1) # Considering 3 features + top60 words (with P)
print ("11 Closed-form: 3f 60wd wP: ")
w = closed_form(TRA(X), TRA(y)) # Only considering 3 features
print ("Train err: ", MSE(TRA(X), TRA(y), w).values, "Valid err: ", MSE(VAL(X), VAL(y), w).values)
X = np.concatenate((X_wP[...,:4], X_wP[...,7:167]),axis=1) # Considering 3 features + top160 words (with P)
print ("13 Closed-form + 3f 160wd wP: ")
w = closed_form(TRA(X), TRA(y)) # Only considering 3 features
print ("Train err: ", MSE(TRA(X), TRA(y), w).values, "Valid err: ", MSE(VAL(X), VAL(y), w).values)
print ("Asked by the instructor, we checked the model and discover that considering top 60 words has better performance on the validation set.")
print ("\n")
#=================================================================
X = np.concatenate((X_noP[...,:4], X_noP[...,7:67]),axis=1) # Considering 3 features + top60 words (without P)
print ("12 Closed-form + 3f 60wd noP: ")
w = closed_form(TRA(X), TRA(y)) # Only considering 3 features
print ("Train err: ", MSE(TRA(X), TRA(y), w).values, "Valid err: ", MSE(VAL(X), VAL(y), w).values)
X = np.concatenate((X_noP[...,:4], X_noP[...,7:167]),axis=1) # Considering 3 features + top160 words (without P)
print ("14 Closed-form + 3f 160wd noP: ")
w = closed_form(TRA(X), TRA(y)) # Only considering 3 features
print ("Train err: ", MSE(TRA(X), TRA(y), w).values, "Valid err: ", MSE(VAL(X), VAL(y), w).values)
print ("Suggested by reference materials, neglecting punctuations might give better results. It turns out that using top 160 words will give better performance in this case.")
print ("\n")
#=================================================================
X = np.concatenate((X_noP[...,:4], x_select_f(X_noP[...,7:167], y)), axis=1)  # Considering 3 features + [top160 words (without P) using feature selection forward]
print ("54 Closed-form + 3f [160wd noP (F)]: ")
w = closed_form(TRA(X), TRA(y)) # Only considering 3 features
print ("Train err: ", MSE(TRA(X), TRA(y), w).values, "Valid err: ", MSE(VAL(X), VAL(y), w).values)
print ("We further implemented word selecting algorithm to process the 160 word features")
print ("\n")
#=================================================================
X = np.concatenate((X_noP[...,:5], x_select_f(X_noP[...,7:167], y)),axis=1) # Considering 3 features + sq + 60noP
print ("25 Closed-form: 3f sq [160wd noP (F)]: ")
w = closed_form(TRA(X), TRA(y)) # Only considering 3 features
print ("Train err: ", MSE(TRA(X), TRA(y), w).values, "Valid err: ", MSE(VAL(X), VAL(y), w).values)
X = np.concatenate((X_noP[...,:6], x_select_f(X_noP[...,7:167], y)),axis=1) # Considering 3 features + sq + 60noP
print ("25 Closed-form: 3f sq log [160wd noP (F)]: ")
w = closed_form(TRA(X), TRA(y)) # Only considering 3 features
print ("Train err: ", MSE(TRA(X), TRA(y), w).values, "Valid err: ", MSE(VAL(X), VAL(y), w).values, "Test err: ", MSE(TEST(X), TEST(y), w).values)
print ("By using basic expansion and transformation of the strong correlated feature 'children', we can get our best prediction.")