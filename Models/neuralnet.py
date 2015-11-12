import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import test1
from test1 import process_data
import numpy as np
import theano


X = np.load('X.npy')
Y = np.load('Y.npy')

print X.shape
print Y.shape

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden1', layers.DenseLayer),
        ('hidden2', layers.DenseLayer),
        ('hidden3', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],

     # layer parameters:
    input_shape=(None, 32),  
    hidden1_num_units=350, 
    hidden2_num_units=350,
    hidden3_num_units=350, 
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=1,  # 30 target values



    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.001,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=400,  # we want to train this many epochs
    verbose=1,
    )


net1.fit(X,Y)

#Load and process test data
test = pd.read_csv('/Users/arjun/Documents/Kaggle/Rosmann/test.csv', parse_dates = ['Date'])
test = process_data(test)

train = pd.read_csv('/Users/arjun/Documents/Kaggle/Rosmann/train.csv', parse_dates = ['Date'])

train= train[train['Open'] != 0]
X_train = process_data(train)

X_train = X_train.drop(['Sales', 'Customers'], axis = 1)



#add placeholder columns to test data
for column in X_train.columns:
    if column not in test.columns:
        test[column] = 0.0
        
test = test.sort_index(axis=1).set_index('Id')
test = test.sort_index(axis=0)



# Standardize test data
test[['CompetitionDistance', 'CompetitionOpen', 'PromoOpen']] = \
                           scaler.transform(test[['CompetitionDistance', 'CompetitionOpen', 'PromoOpen']])



# Convert to numpy array
test_X = test.values

# Covert test data to float32
test_X = test_X.astype(np.float32)

# Make prediction
y_test = net1.predict(test_X)

# Inverse transform after prediction:
y_test = scaler_y.inverse_transform(y_test)

# Make submission
result = pd.DataFrame({'Id': test.index.values, 'Sales': y_test}).set_index('Id')
result.to_csv('submission_nn1.csv')
print('submission created')







