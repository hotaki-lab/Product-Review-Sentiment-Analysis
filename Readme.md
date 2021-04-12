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

# Defining a Hinge Loss Function:

In this project you will be implementing linear classifiers beginning with the Perceptron algorithm. You will begin by writing your loss function, a hinge-loss function. For this function you are given the parameters of your model θ and θ0. Additionally, you are given a feature matrix in which the rows are feature vectors and the columns are individual features, and a vector of labels representing the actual sentiment of the corresponding feature vector.

### Hinge Loss Function for One Data Sample:

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

### Hinge Loss Function for a Complete Dataset:

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

# (a) Applying the Perceptron Algorithm:

### Perceptron Single Step Update:

Now you will implement the single step update for the perceptron algorithm (implemented with 0−1 loss). You will be given the feature vector as an array of numbers, the current θ and θ_0 parameters, and the correct label of the feature vector. The function should return a tuple in which the first element is the correctly updated value of θ and the second element is the correctly updated value of θ_0.

```
def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a single step of the perceptron algorithm.
    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.
    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    
    tx = np.dot(current_theta, feature_vector)
    txtz = tx + current_theta_0
    ytx = label * tx
    ytxtz = label * txtz

    if ytxtz <= 0:
        current_theta += label * feature_vector
        current_theta_0 += label
    return (current_theta, current_theta_0)
        
    raise NotImplementedError
```

### Applying Full Perceptron Algorithm:

In this step you will implement the full perceptron algorithm. You will be given the same feature matrix and labels array as you were given in The Complete Hinge Loss. You will also be given T, the maximum number of times that you should iterate through the feature matrix before terminating the algorithm. Initialize θ and θ_0 to zero. This function should return a tuple in which the first element is the final value of θ and the second element is the value of θ0.

Tip: Call the function **perceptron_single_step_update** directly without coding it again.

**Hint: Make sure you initialize theta to a 1D array of shape (n,) and not a 2D array of shape (1, n).**

Note: Please call get_order(feature_matrix.shape[0]), and use the ordering to iterate the feature matrix in each iteration. The ordering is specified due to grading purpose. In practice, people typically just randomly shuffle indices to do stochastic optimization.

```
def perceptron(feature_matrix, labels, T):
    
    """
    Runs the full perceptron algorithm on a given set of data. Runs iterations through the data set, there is no need to worry about stopping early.
    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.
    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])
    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row represents a single data point. 
        labels - A numpy array where the kth element of the array is the correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm should iterate through the feature matrix.
    Returns: A tuple where the first element is a numpy array with the value of theta, the linear classification parameter, after T iterations through the feature matrix and the second element is a real number with the value of theta_0, the offset classification parameter, after T iterations through the feature matrix.
    """
    # Your code here
    (n, k) = feature_matrix.shape
    theta = np.zeros(k)
    theta_0 = 0.0

    for t in range(T):
        for i in get_order(n):
            # Your code here
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
    return (theta, theta_0)
        
    raise NotImplementedError
```

### Testing Full Perceptron on a Toy Dataset:
![alt text](https://github.com/hotaki-lab/Product-Review-Sentiment-Analysis/blob/main/Figure_1%20(perceptron).png "Perceptron Application on Toy Dataset")

# (b) Average Perceptron Algorithm:

The average perceptron will add a modification to the original perceptron algorithm: since the basic algorithm continues updating as the algorithm runs, nudging parameters in possibly conflicting directions, it is better to take an average of those parameters as the final answer. Every update of the algorithm is the same as before. The returned parameters θ, however, are an average of the θs across the nT steps:

**θ_final = (1 / nT) (θ1 + θ2 + ... + θnT)**

You will now implement the average perceptron algorithm. This function should be constructed similarly to the Full Perceptron Algorithm above, except that it should return the average values of θ and θ0

Tip: Tracking a moving average through loops is difficult, but tracking a sum through loops is simple.

Note: Please call get_order(feature_matrix.shape[0]), and use the ordering to iterate the feature matrix in each iteration. The ordering is specified due to grading purpose. In practice, people typically just randomly shuffle indices to do stochastic optimization.

```
def average_perceptron(feature_matrix, labels, T):
    
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.
    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.
    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])
    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row represents a single data point. 
        labels - A numpy array where the kth element of the array is the correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm should iterate through the feature matrix.
    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
   
    (n, k) = feature_matrix.shape
    theta = np.zeros(k)
    theta_total = np.zeros(k)
    theta_0 = 0.0
    theta_0_total = 0.0

    for t in range(T):
        for i in get_order(n):

            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            theta_total += theta
            theta_0_total += theta_0
    return (theta_total / (n*T), theta_0_total / (n*T))
    
    raise NotImplementedError
```

### Testing Avg. Perceptron on a Toy Dataset:
![alt text](https://github.com/hotaki-lab/Product-Review-Sentiment-Analysis/blob/main/Figure_1%20(avg%20perceptron).png "Avg. Perceptron Application on Toy Dataset")

# PEGASOS Algorithm:

Now you will implement the Pegasos algorithm. For more information, refer to the original paper at original paper.

The following pseudo-code describes the Pegasos update rule.

![alt text](https://github.com/hotaki-lab/Product-Review-Sentiment-Analysis/blob/main/PEGASOS%20Equation.JPG "PEGASOS Equation")

### Pegasos Single Step Update:

Next you will implement the single step update for the Pegasos algorithm. This function is very similar to the function that you implemented in Perceptron Single Step Update, except that it should utilize the Pegasos parameter update rules instead of those for perceptron. The function will also be passed a λ and η value to use for updates.

```
def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
   
   """
    Properly updates the classification parameter, theta and theta_0, on a single step of the Pegasos algorithm
    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos algorithm before this update.
        current_theta_0 - The current theta_0 being used by the Pegasos algorithm before this update.
    Returns: A tuple where the first element is a numpy array with the value of theta after the current update has completed and the second element is a real valued number with the value of theta_0 after the current updated has completed.
    """
    
    tx = np.dot(current_theta, feature_vector)
    ytx = label * tx
    txtz = tx + current_theta_0
    ytxtz = label * txtz
    coef = 1-eta*L
    
    if ytxtz <= 1:
        current_theta = (coef)*(current_theta) + (eta * label * feature_vector)
        current_theta_0 += eta * label
    else:
        current_theta = (coef)*(current_theta)

    return (current_theta, current_theta_0)
    
    raise NotImplementedError
```

### Full Pegasos Algorithm:

Finally, you will implement the full Pegasos algorithm. You will be given the same feature matrix and labels array as you were given in Full Perceptron Algorithm. You will also be given T, the maximum number of times that you should iterate through the feature matrix before terminating the algorithm. Initialize θ and θ0 to zero. For each update, set η=1t√ where t is a counter for the number of updates performed so far (between 1 and nT inclusive). This function should return a tuple in which the first element is the final value of θ and the second element is the value of θ0.

Note: Please call get_order(feature_matrix.shape[0]), and use the ordering to iterate the feature matrix in each iteration. The ordering is specified due to grading purpose. In practice, people typically just randomly shuffle indices to do stochastic optimization.

```
def pegasos(feature_matrix, labels, T, L):
   
   """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.
    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).
    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.
    Args:
        feature_matrix - A numpy matrix describing the given data. Each row represents a single data point.
        labels - A numpy array where the kth element of the array is the correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos algorithm parameters.
    Returns: A tuple where the first element is a numpy array with the value of the theta, the linear classification parameter, found after T iterations through the feature matrix and the second element is a real number with the value of the theta_0, the offset classification parameter, found after T iterations through the feature matrix.
    """

    (nsamples, nfeatures) = feature_matrix.shape
    theta = np.zeros(nfeatures)
    theta_0 = 0
    count = 0
    for t in range(T):
        for i in get_order(nsamples):
            count += 1
            eta = 1.0 / np.sqrt(count)
            (theta, theta_0) = pegasos_single_step_update(
            feature_matrix[i], labels[i], L, eta, theta, theta_0)
    return (theta, theta_0)
    raise NotImplementedError
```

![alt text](https://github.com/hotaki-lab/Product-Review-Sentiment-Analysis/blob/main/Figure_1%20(pegasos)%20.png "Full PEGASOS Application on a Toy Dataset")

# Convergence of 3 Algorithms Explained Above:

Since you have implemented three different learning algorithm for linear classifier, it is interesting to investigate which algorithm would actually converge. Please run it with a larger number of iterations T to see whether the algorithm would visually converge. You may also check whether the parameter in your theta converge in the first decimal place. Achieving convergence in longer decimal requires longer iterations, but the conclusion should be the same.

**The following algorithms will converge on this dataset: **
** 1. average perceptron algorithm
   2. pegasos algorithm**


# Sentiment Analysis Project (Product Review Analysis):

Now that you have verified the correctness of your implementations, you are ready to tackle the main task of this project: building a classifier that labels reviews as positive or negative using text-based features and the linear classifiers that you implemented in the previous section!

### The Data:

The data consists of several reviews, each of which has been labeled with −1 or +1, corresponding to a negative or positive review, respectively. The original data has been split into four files:

**reviews_train.tsv (4000 examples)
reviews_validation.tsv (500 examples)
reviews_test.tsv (500 examples)**

To get a feel for how the data looks, we suggest first opening the files with a text editor, spreadsheet program, or other scientific software package (like pandas).
Translating reviews to feature vectors
We will convert review texts into feature vectors using a bag of words approach. We start by compiling all the words that appear in a training set of reviews into a dictionary , thereby producing a list of d unique words.

We can then transform each of the reviews into a feature vector of length d by setting the ith coordinate of the feature vector to 1 if the ith word in the dictionary appears in the review, or 0 otherwise. For instance, consider two simple documents “Mary loves apples" and “Red apples". In this case, the dictionary is the set {Mary;loves;apples;red}, and the documents are represented as (1;1;1;0) and (0;0;1;1).

A **bag of words model** can be easily expanded to include phrases of length m. A unigram model is the case for which m=1. In the example, the unigram dictionary would be (Mary;loves;apples;red). In the bigram case, m=2, the dictionary is (Mary loves;loves apples;Red apples), and representations for each sample are (1;1;0),(0;0;1). In this section, you will only use the unigram word features. These functions are already implemented for you in the bag of words function.
In utils.py, we have supplied you with the load data function, which can be used to read the .tsv files and returns the labels and texts. We have also supplied you with the bag_of_words function in project1.py, which takes the raw data and returns dictionary of unigram words. The resulting dictionary is an input to extract_bow_feature_vectors which computes a feature matrix of ones and zeros that can be used as the input for the classification algorithms. Using the feature matrix and your implementation of learning algorithms from before, you will be able to compute θ and θ0.

