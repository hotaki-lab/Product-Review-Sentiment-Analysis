# Introduction & Scope of Project (Sentiment Analysis):
The goal of this project is to design a classifier to use for **sentiment analysis of product reviews**. Our training set consists of reviews written by **Amazon** customers for various food products. The reviews, originally given on a 5 point scale, have been adjusted to a +1 or -1 scale, representing a positive or negative review, respectively.

Below are two example entries from our dataset. Each entry consists of the review and its label. The two reviews were written by different customers describing their experience with a sugar-free candy.

Colons can be used to align columns.

|                                                 **Review Sample**                                         |   **Label**   |
| --------------------------------------------------------------------------------------------------------- |:-------------:|
| Nasty No flavor. The candy is just red, No flavor. Just plan and chewy. I would never buy them again      |       -1      |
| YUMMY! You would never guess that they're sugar-free and it's so great that you can eat them pretty much guilt free! i was so impressed that i've ordered some for myself (w dark chocolate) to take to the office. These are just EXCELLENT! |       +1      |

In order to automatically analyze reviews, the following tasks to be completed first:

1. Implement and compare three types of **linear classifiers**: (a) **The Perceptron Algorithm,** (b) **The Average Perceptron Algorithm,** and (c) **The PEGASOS Algorithm**.

2. Use your classifiers on the food review dataset, using some simple text features.

3. Experiment with additional features and explore their impact on classifier performance.

# Setup Details:

For this project and throughout the project we will be using **Python 3.6** with some additional libraries. We strongly recommend that you take note of how the NumPy numerical library is used in the code provided, and read through the on-line NumPy tutorial. NumPy arrays are much more efficient than Python's native arrays when doing numerical computation. In addition, using NumPy will substantially reduce the lines of code you will need to write.

Note on software: For this project, you will need the NumPy numerical toolbox, and the matplotlib plotting toolbox.

Download **sentiment_analysis.tar.gz** and untar it into a working directory. The sentiment_analysis folder contains the various data files in .tsv format, along with the following python files:

project1.py contains various useful functions and function templates that you will use to implement your learning algorithms.

main.py is a script skeleton where these functions are called and you can run your experiments.

utils.py contains utility functions that the staff has implemented for you.

test.py is a script which runs tests on a few of the methods you will implement. Note that these tests are provided to help you debug your implementation and are not necessarily representative of the tests used for online grading. Feel free to add more test cases locally to further validate the correctness of your code before submitting to the online graders in the codeboxes.

# Hinge Loss:

In this project you will be implementing linear classifiers beginning with the Perceptron algorithm. You will begin by writing your loss function, a hinge-loss function. For this function you are given the parameters of your model θ and θ0. Additionally, you are given a feature matrix in which the rows are feature vectors and the columns are individual features, and a vector of labels representing the actual sentiment of the corresponding feature vector.

## Hinge Loss Function for One Data Sample:

First, implement the basic hinge loss calculation on a single data-point. Instead of the entire feature matrix, you are given one row, representing the feature vector of a single data sample, and its label of +1 or -1 representing the ground truth sentiment of the data sample.

Reminder: You can implement this function locally first, and run python test.py in your sentiment_analysis directory to validate basic functionality.

```python
def hinge_loss_single(feature_vector, label, theta, theta_0):
   
   """
    Finds the hinge loss on a single data point given specific classification parameters.
    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.
    Returns: A real number representing the hinge loss associated with the given data point and parameters.
    """
    
    z = label*(theta.dot(feature_vector)+theta_0)
    singleHingeLoss = 1-z
    
    if z >= 1:
        singleHingeLoss = 0
    
    return singleHingeLoss
    
    raise NotImplementedError
```

## Hinge Loss Function for a Complete Dataset:

Now it's time to implement the complete hinge loss for a full set of data. Your input will be a full feature matrix this time, and you will have a vector of corresponding labels. The k-th row of the feature matrix corresponds to the k-th element of the labels vector. This function should return the appropriate loss of the classifier on the given dataset.

```python
def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.
    Args:
        feature_matrix - A numpy matrix describing the given data. Each row represents a single data point.
        labels - A numpy array where the kth element of the array is the correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.
    Returns: A real number representing the hinge loss associated with the given dataset and parameters. This number should be the average hinge loss across all of the points in the feature matrix.
    """
    
    Z = 0
    for i in range(len(feature_matrix)):
        Z += hinge_loss_single(feature_matrix[i], labels[i], theta, theta_0)
    return Z / len(labels)

    raise NotImplementedError
```
