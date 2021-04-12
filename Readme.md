# Introduction:
The goal of this project is to design a classifier to use for **sentiment analysis of product reviews**. Our training set consists of reviews written by **Amazon** customers for various food products. The reviews, originally given on a 5 point scale, have been adjusted to a +1 or -1 scale, representing a positive or negative review, respectively.

Below are two example entries from our dataset. Each entry consists of the review and its label. The two reviews were written by different customers describing their experience with a sugar-free candy.

Colons can be used to align columns.

|                                                      **Review**                                           |   **Label**   |
| --------------------------------------------------------------------------------------------------------- |:-------------:|
| Nasty No flavor. The candy is just red, No flavor. Just plan and chewy. I would never buy them again      |       -1      |
| YUMMY! You would never guess that they're sugar-free and it's so great that you can eat them pretty much guilt free! i was so impressed that i've ordered some for myself (w dark chocolate) to take to the office. These are just EXCELLENT! |       +1      |

In order to automatically analyze reviews, the following tasks to be completed first:

1. Implement and compare three types of **linear classifiers**: (a) **The Perceptron Algorithm,** (b) **The Average Perceptron Algorithm,** and (c) **The PEGASOS Algorithm**.

2. Use your classifiers on the food review dataset, using some simple text features.

3. Experiment with additional features and explore their impact on classifier performance.
