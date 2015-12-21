# Enron Email Scandal Investigation
===================================================
By: Jayadeep Jayaraaman
@jayadeepjayaraman

## Introduction
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for to executives.

From a $90 price per share, to a $1 value represents the huge value loss and scam that happened in Enron. This case has been
a point of interest for machine learning analysis because of the huge real-world impact that ML could help out and try to figure out what went wrong and how to avoid it in the future. It would be of great value to find a model that could potentially predict these types of events before much damage is done, so as to permit preventive action. Corporate governance, the stock market, and even the Government would be quite interested in a machine learning model that could signal potential fraud detections before hand.

Utilizing `scikit-learn` and machine learning methodologies, I built a "person of interest" (POI) identifier to detect and predict culpable persons, using features from financial data, email data, and labeled data--POIs who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

Below are some questions and their answers to understand the data and the machine learning techniques used better

> Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?

## Enron Data
The interesting and hard part of the dataset is that the distribution of the non-POI's to POI's is very skewed, given that from the 146 there are only 11 people or data points labeled as POI's or guilty of fraud. We are interested in labeling every person in the dataset into either a POI or a non-POI (POI stands for *Person Of Interest*). 

## Data Processing

In total I have used 24 features, out of which 14 features are present in the input dataset and 10 new features were created.

- poi_ratio_messages - This feature calculates ratio of the messages exchanged with Person of Interest and the Total Messages
- Log Transformation - Few of the numerical values were converted to a corresponding log values
- Squared Transformation - Few of the numerical values were converted to a corresponding squared values

Log and Squared transformed values were created because sometimes the numerical data is normally distributed when they are transformed that way.

As the data is very sparse I didn't filter any data which was NaN except for the Outlier removal as explained below.

There are 2 clear outliers in the data, **TOTAL** and **THE TRAVEL AGENCY IN THE PARK**. The first one seems to be the sum total of all the other data points, while the second outlier is quite bizarre. Both these outliers are removed from the dataset for all the analysis. 

The first outlier **TOTAL** was found by plotting the value and finding the outlier.
The second outlier **THE TRAVEL AGENCY IN THE PARK** was manually reviewing the pdf document.

> What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values

### New features
From the initial dataset, 5 new features where added, you can find more details in the table below:

|Feature | Description     |
|--------|-----------------|
|Ratio of POI messages | POI related messages divided over the total messages from the person |
|Log of financials (multiple) | Financial variables with logarithmic transformation |
|Square of financials (multiple) | Financial variables with squared transformation |

The reason behind the **ratio of POI messages** is that we expect that POI's contact each other relatively more often than with non-POI's. I also can expect that the financial gains to be non linear and therefore it will show up normally distributed in a logarithmic scale and/or squared transformation. This might help the algorithms.

### Feature Scaling
I scaled the features using the MinMaxScalaer() as there are different kinds of numerical data following different scales.

### PCA
I didnt spend too much time in using K-Best or any other feature reduction/selection techniques, the reason being I wanted to use PCA as a pre-processing step to one of the classifiers. It's quite reasonable to think that all the **email** features we have, 5 initial features plus 1 computed feature, really represent 1 underlying feature or principal component, something like increased amount of communication between POI's versus between POI's and non-POI's. The same goes for the financial features, which we could think are really measuring the POI's corruption via big money gains. In other words, we expect that a POI has a higher money gain compared to a non-POI, and that all the financial features are really trying to measure this underlying one. By tuning the parameters, we get the best classification results and from the **29** features in total, they are reduced to **20** principal components.

> What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

## Algorithms selection and tuning
For the analysis of the data, a total of 10 classifiers were tried out, which include:
- Logistic Regression
- Linear Discriminant Analysis
- Decision Tree Classifier
- Gaussian Naive Bayes
- Linear Support Vector Classifier (LinearSVC)
- AdaBoost
- Random Forrest Tree Classifier
- K Nearest Neighbor
- KMeans
- Bernoulli RBM (together with Logistic Regression)

The object of the algorithm is to classify and find out which people are more likely to be POI's. There are clearly
2 categories we are looking to label the data.

> What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm?

To tune the overall performance, both automated and manual tuning of parameters was involved. The automated tuned parameters where done using the **GridSearchCV** from SkLearn. The manual tuning occurred in the following ways:
* 1. Including the PCA features
* 2. Adding/removing features
* 3. Scaling features

For the most part, PCA made a huge improvement when the new features where added. PCA is kind of getting the best parts of the 29 features and cramming them up into 20. The new features really made the difference to push recall and precision up.


### Optimization
All the machine-learning algorithms where optimized using **GridSearchCV**. The general
process was:
* > build list of classifier with parameters 
* > optimize each classifier with training data 
* > evaluate all the classifiers 
* > compare f1, recall and precision scores 
* > choose the best classifier

> What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?

### Validation and Performance
To validate the performance of each algorithm, `recall`, `precision` and `F1 scores` where calculated for each one. You can find below a summary of the scores of the top algorithms.

|Feature | F1 Score | Recall | Precision | Accuracy |
|--------|----------|--------|-----------|-----------|
|Logistic Regression          |  0.42666 | 0.50100 | 0.37152 |0.82047|
|BernoulliRBM|0.39213|0.69250|0.27350|0.71373|

The best classifier was actually a *Logistic Regression* using PCA and scaling beforehand. This was achieved by using `sklearn Pipline`. The logistic regression achieved a consistent score above 0.30 for both precision and recall. 

It seems that the most important parameter to tune was to set the `class_weight` to `auto`. I suspect this is due to the skewed nature of the dataset, because class weight assigns the importance of each class (POI or non-POI) depending on the inverse appearance of the class. So it set a much higher importance to POI's class which is exactly what we want in this case.

The main evaluation metrics utilized were precision and recall. Precision captures the ratio of true positives to the records that are actually POIs, essentially describing how often 'false alarms' are (not) raised. Recall captures the ratio of true positives to the records flagged as POIs, which describes sensitivity. Due to the unbalanced nature of the dataset (few POIs), accuracy is certainly not a good metric, i.e. if 'non-POI' had been predicted for all records, an accuracy of 82.04% would have been achieved. 

Given the context of assisting and enabling securities and fraud investigators, I would argue that precision is secondary to recall. Simply put, with the objective of 'flagging' individuals for further human-led investigation, it is more important that suspect individuals are included than innocent individuals be excluded. A high recall value would ensure that truly culpable individuals were flagged as POIs and would be investigated more thoroughly.

## Discussion and Conclusions
The most challenging aspect of this project was the sparse nature of the dataset, with very few (18) POIs. Most of the algorithms employed perform much better in balanced datasets. An interesting next step would be to employ anomaly detection algorithms (as used in other cases of fraud, like credit card checking) to identify persons of interest. 

## References
- [Udacity - Intro to Machine Learning course](https://www.udacity.com/course/ud120)
- [Sklearn documentation](http://scikit-learn.org/stable/documentation.html)
