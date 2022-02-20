# Sentiment-Analyzer
MITx - MicroMasters Program on Statistics and Data Science - Machine Learning with Python 

First Project - Linear Classifier for Sentiment Analysis

The first project of the MIT MicroMasters Program course on Machine Learning with Python consisted of designing a linear classifier for sentiment analysis of product reviews. This meant, coding from scratch several linear classifiers, the loss functions for these classifiers and finally the metrics to analyze the results. 

The training set consists of reviews written by Amazon customers for various food products that have been adjusted from the original 5 point scale to a +1 -1 scale to represent a positive or negative review. The training set has 4000 reviews, validation set 500 reviews and the test set a total of 500 reviews.

The goal of the project was first to implement and compare three types of linear classifiers: the Perceptron algorithm, the Average Percetron algorithm, and the Pegasus algorithm. Secondly, to use these classifiers on the food review dataset by using some simple text features, and finally, to experiment with additional features and explore their impact on classifier performance.

Additional helper functions were given to complete the project in two weeks of time.

**ACCESSING CODE**

The file main.py runs the code with the help of the two modules (project1.py and utils.py) that contain helper and utility functions.

The dependencies and requirements can be seen from requirements.txt that can be installed in shell with the command:

      pip install -r requirements.txt
