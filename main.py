import argparse
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import MultiLayerPerceptron
from dataset import AdultDataset
from util import *

""" Adult income classification

In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

"""
# =================================== LOAD DATASET =========================================== #

######

data = pd.read_csv(r'adult.csv')

######

# =================================== DATA VISUALIZATION =========================================== #
######

print("Shape of our dataset is " + str(data.shape))
print(data.head())
print("Names of all columns " + str(data.columns.to_list()))

print("Number of low earners: " + str(data.income.value_counts()[0]))
print("Number of high earners: " + str(data.income.value_counts()[1]))

######


# =================================== DATA CLEANING =========================================== #

# datasets often come with missing or null values, this is an inherent limit of the data collecting process
# before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
# detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
# indicated with the symbol "?" in the dataset

# let's first count how many missing entries there are for each feature
col_names = data.columns
num_rows = data.shape[0]
for feature in col_names:
    print(str(feature) + " : " + str(data[feature].isin(["?"]).sum()))

# next let's throw out all rows (samples) with 1 or more "?"


# So only workclass, occupation, and native-country have missing values
data = data[data["workclass"] != "?"]
data = data[data["occupation"] != "?"]
data = data[data["native-country"] != "?"]


# =================================== BALANCE DATASET =========================================== #

data_low_earner = data[data["income"] == "<=50K"]  # Separate dataframe into both classes
data_high_earner = data[data["income"] == ">50K"]

data_low_earner = data_low_earner.sample(n=data_high_earner.shape[0], random_state=1)  # sample to get same amount

data = pd.concat([data_low_earner, data_high_earner], axis=0)  # Rejoin the dataframes by row
data = data.sample(frac=1, random_state=1).reset_index(drop=True)  # reshuffle dataset and drop new column of index labelling that is made

# =================================== DATA STATISTICS =========================================== #

# our dataset contains both continuous and categorical features. In order to understand our continuous features better,
# we can compute the distribution statistics (e.g. mean, variance) of the features using the .describe() method

######
print(data.describe())

# Seeing how many male vs females are high earners
print(((data["gender"] == "Male") & (data["income"] == ">50K")).sum())
print(((data["gender"] == "Female") & (data["income"] == ">50K")).sum())
print((data["income"] == ">50K").sum())

######

# likewise, let's try to understand the distribution of values for discrete features. More specifically, we can check
# each possible value of a categorical feature and how often it occurs
categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                     'relationship', 'gender', 'native-country', 'income']


for feature in categorical_feats:
    #####
    pie_chart(data, feature)  # Make a piechart of categorical features

    ######

# visualize the first 3 features using pie and bar graphs

for feature in categorical_feats:
    binary_bar_chart(data, feature)

######

# =================================== DATA PREPROCESSING =========================================== #

# we need to represent our categorical features as 1-hot encodings
# we begin by converting the string values into integers using the LabelEncoder class
# next we convert the integer representations into 1-hot encodings using the OneHotEncoder class
# we don't want to convert 'income' into 1-hot so let's extract this field first
# we also need to preprocess the continuous features by normalizing against the feature mean and standard deviation
# don't forget to stitch continuous and cat features together

# NORMALIZE CONTINUOUS FEATURES
# Make a list of all continous features
cont_feats = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
             'relationship', 'gender', 'native-country', 'income']

data_cont = data[cont_feats]  # 2 new dataframes for continuous and categorical features
data_categorical = data[cat_feats]

normalized_data = data_cont.copy()

for feature in cont_feats:
    mean = data_cont[feature].mean()
    std = data_cont[feature].std()
    normalized_data[feature] = (data_cont[feature] - mean) / std

data_cont = normalized_data
data_cont = data_cont.values  # Turn it into a numpy array

# ENCODE CATEGORICAL FEATURES
label_encoder = LabelEncoder()
encoded_data = data_categorical.copy()

for feature in cat_feats:
    label_encoder.fit(data_categorical[feature])
    encoded_data[feature] = label_encoder.transform(data_categorical[feature])

data_categorical = encoded_data

data_income = data_categorical["income"]  # This is now it's own column
data_income = data_income.values  # Turn into a numpy array
data_categorical.drop('income', axis=1, inplace=True)  # get rid of income column from dataframe
######

oneh_encoder = OneHotEncoder()

onehot_categorical_data = oneh_encoder.fit_transform(data_categorical)  # returns a numpy array

onehot_categorical_data = onehot_categorical_data.toarray()

# Put everything together into one numpy array with all the data
processed_data = np.concatenate((data_cont, onehot_categorical_data), axis=1)

# =================================== MAKE THE TRAIN AND VAL SPLIT =========================================== #
# we'll make use of the train_test_split method to randomly divide our dataset into two portions
# control the relative sizes of the two splits using the test_size parameter

X_train, X_test, y_train, y_test = train_test_split(processed_data, data_income, test_size=0.20, random_state=1)


######

# =================================== LOAD DATA AND MODEL =========================================== #

def load_data(batch_size):

    training_set = AdultDataset(X_train, y_train)
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

    validation_set = AdultDataset(X_test, y_test)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def load_model(lr):

    model = MultiLayerPerceptron(103)
    loss_fnc = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer


def evaluate(model, val_loader):
    total_corr = 0

    for batch_index, data_load in enumerate(val_loader):
        val_data = data_load[0].float()
        val_label = data_load[1].float().squeeze()

        prediction = model(val_data)
        prediction = np.where(prediction >= 0.5, 1, 0)
        for k in range(0, len(prediction), 1):
            if prediction[k] == val_label[k].item():
                total_corr += 1.0

    return float(total_corr) / len(val_loader.dataset)


def num_correct(prediction, label):
    num_corr = 0
    prediction = np.where(prediction >= 0.5, 1, 0)
    for k in range(0, len(label), 1):
        if prediction[k] == label[k].item():
            num_corr += 1.0
    return num_corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=10)

    args = parser.parse_args()
    ######

    train_loader, val_loader = load_data(args.batch_size)
    model, criterion, optimizer = load_model(args.lr)

    # Now we create some arrays to store data from training so we can plot them
    num_training_batches = len(train_loader.dataset) / args.batch_size
    num_validation_batches = len(val_loader.dataset) / args.batch_size

    training_accuracy = [0] * (int(((args.epochs) * (num_training_batches) / args.eval_every)) + 1)
    validation_accuracy = [0] * (int(((args.epochs) * (num_training_batches) / args.eval_every)) + 1)
    gradient_steps_array_train = [0] * (
                int(((args.epochs) * (num_training_batches) / args.eval_every)) + 1)  # Just to graph something against

    for i in range(0, (int(((args.epochs) * (num_training_batches) / args.eval_every)) + 1), 1):
        gradient_steps_array_train[i] = i

    num_correct_pred = 0
    num_N_steps = 0
    for i in range(0, args.epochs, 1):
        for batch_index, data_load in enumerate(train_loader):
            # batch_index is what batch we're on ofc
            # data is a tuple with first one being training tensors
            # and then another tensor of the labels
            optimizer.zero_grad()
            input_data = data_load[0].float()  # torch.Size([15, 103])
            input_label = data_load[1].float()  # torch.Size([15])

            predictions = model(input_data)
            loss = criterion(predictions.squeeze(), input_label.squeeze())

            loss.backward()
            optimizer.step()
            if batch_index % (args.eval_every) == 0:
                num_correct_pred += num_correct(predictions,
                                                input_label.squeeze())  # For all minibatches evaluated, how many were correct
                num_N_steps += 1  # How many times we've stopped here to evaluate
                num_batches_so_far = i * num_training_batches + batch_index
                training_accuracy[num_N_steps - 1] = (num_correct_pred / (
                            num_N_steps * args.batch_size))  # num_N_steps * args.batch_size is how many samples were in the N stops we had so far
                validation_acc = evaluate(model, val_loader)
                validation_accuracy[num_N_steps - 1] = (validation_acc)
                print("epoch : " + str(i) + " batch: " + str(batch_index) + " loss: " +
                      str(loss) + "training acc: " + str(
                            num_correct_pred / (num_N_steps * args.batch_size)) + " valid acc: " + str(validation_acc))

    ######
    font = {'size': 6}
    plt.figure(1)  # Figure 1 is for training and validation loss + accuracy
    plt.subplot(211)
    plt.plot(gradient_steps_array_train, training_accuracy, 'r', gradient_steps_array_train, validation_accuracy, 'g')
    plt.xlabel('Gradient Steps')
    plt.ylabel('accuracy')
    plt.title('Training and Validation accuracy by minibatch gradient step. learning rate: ' + str(
        args.lr) + " epochs: " + str(args.epochs) + " batch size: " + str(args.batch_size) + " and 50 hidden neurons",
              fontdict=font)
    plt.legend(['training accuracy', 'validation accuracy'])
    plt.show()

if __name__ == "__main__":
    main()
