# PLAYER STRATEGY CLASSIFICATION USING ARTIFICIAL NEURAL NETWORKS AND LOGISTIC REGRESSION
This project addresses the classification of player strategy based on the feature vector acquired from FIFA`s official gaming website. The Logistic Regression is used with a sigmoid function for finding the performance of the model. Upon finding over-fitted results, the ANN model is taken into consideration for evaluating the better performance of the model. The Keras is used for deploying the ANN model.

<b>By Saif Mathur</b>
<h2>INTRODUCTION</h2>
<p>The dataset used here is historical data from FIFA`s official gaming
website. The objective of this work is to classify the strategy of a game
plan by an individual player i.e. whether the player should be placed in
the ‘Attack Position’ or ‘Defence Position’ .

The input is a tuple of different attributes of the player in integer form
submitted by the user. For example the user will be asked to provide player
specification, the player`s aggression level or player`s accuracy.

As the standard procedure the dataset was cleaned first, replacing missing
values with their respective median and manually replaced by checking
player statistics from the same website. Also some attributes were removed
as they were irrelevant, for example the position of a goal keeper since we
are only classifying players into ‘Attack’ or ‘Defence’.

The columns were then rearranged for better understanding.

The dataset was then divided into two sets to train and test the model.

Initially, the algorithm of Logistic Regression is applied, since this model
is a linear model it could not find a clear boundary and gave poor results.

The second algorithm applied on the same dataset is an Artificial Neural
Network which gave a better accuracy.</p>
