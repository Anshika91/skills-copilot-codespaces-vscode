import pandas
get_ipython().system('pip install statsmodels')
get_ipython().system('wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/graphing.py')
get_ipython().system('wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/avalanche.csv')
import graphing # custom graphing code. See our GitHub repo for details

#Import the data from the .csv file
dataset = pandas.read_csv('avalanche.csv', delimiter="\t")

#Let's have a look at the data and the relationship we're going to model
print(dataset.head())

graphing.box_and_whisker(dataset, label_x="avalanche", label_y="weak_layers")
from sklearn.model_selection import train_test_split

# Split the dataset in an 75/25 train/test ratio. 
train, test = train_test_split(dataset, test_size=0.25, random_state=10)

print("Train size:", train.shape[0])
print("Test size:", test.shape[0])


# ## Fitting a model
import statsmodels.formula.api as smf

# Perform logistic regression.
model = smf.logit("avalanche ~ weak_layers", train).fit()

print("Model trained")


# ## Assessing the model with summary information
model.summary()

# ## Assessing model visually

def predict(weak_layers):
    return model.predict(dict(weak_layers=weak_layers))

graphing.scatter_2D(test, label_x="weak_layers", label_y="avalanche", trendline=predict)
graphing.scatter_2D(test, label_x="weak_layers", label_y="avalanche", x_range=[-20,20], trendline=predict)

# ## Assess with cost function
from sklearn.metrics import log_loss

# Make predictions from the test set
predictions = model.predict(test)

# Calculate log loss
print("Log loss", log_loss(test.avalanche, predictions))

# ## Assess accuracy
import numpy

# Print a few predictions before we convert them to categories
print(f"First three predictions (probabilities): {predictions.iloc[0]}, {predictions.iloc[1]}, {predictions.iloc[2]}")

# convert to absolute values
avalanche_predicted = predictions >= 0.5

# Print a few predictions converted into categories
print(f"First three predictions (categories): {avalanche_predicted.iloc[0]}, {avalanche_predicted.iloc[1]}, {avalanche_predicted.iloc[2]}")


# Now we can calculate accuracy:
 
guess_was_correct = test.avalanche == avalanche_predicted
accuracy = numpy.average(guess_was_correct)

# Print the accuracy
print("Accuracy for whole test dataset:", accuracy)
false_negative = numpy.average(numpy.logical_not(guess_was_correct) & test.avalanche)

# False positive: calculate how often it guessed avalanche, when none actually happened
false_positive = numpy.average(numpy.logical_not(guess_was_correct) & numpy.logical_not(test.avalanche))
print(f"Wrongly predicted an avalanche {false_positive * 100}% of the time")
print(f"Failed to predict avalanches {false_negative * 100}% of the time")
