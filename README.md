<h1 align = center>Credit Card Fraud Detection Using Support Vector Machines</h1>

## Requirements:
- pandas
- numpy
- sklearn
- matplotlib
- seaborn

In this project we try to detect credit card fraud using Support Vector Machine also we preprocessing the data.

Database used is [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle

## Data Visualzation
We start by loading the data into the jupyter notebook. After loading the data, we convert the data into a data frame using the pandas to make it more easier to handel.
After loading the data, we visualize the data. First we need to know how our data looks so we use `dataframe.head()` to visualize the first 5 rows of the data also we need to know how our data is distributed so we plot our data.

<p align='center'><img src = 'https://user-images.githubusercontent.com/54438860/119931359-19632800-bf36-11eb-949e-3318c7e9fe54.png'></p>
<h5 align = 'center'> Fig 1: Frauds hppened with respect to the time frame and their respective amounts.</h5>

### Correlation of features
Using `dataframe.corr()`, we find the Pearson, Standard Correlation Coefficient matrix.
<p align = 'center'><img src = 'https://user-images.githubusercontent.com/54438860/119931225-d7d27d00-bf35-11eb-81e4-6bad164137ab.png'></p>
<h5 align = 'center'>Fig 2: Correlation of the futures</h5>

## Data Selection
Since the data is `highly Unbalanced` We need to undersample the data.

**Why are we undersampling instead of oversampling?**

We are undersampling the data because our data is highly unbalanced. The number of transactions which are not fradulent are labeled as 0 and the trancactions whoch are fradulent are labeled as 1.

The number of non fraudulent transactions are **284315** and the number of fradulent transactions are **492**.

If we oversample our data so inclusion of almost **284000** dummy elements will surely affect our outcome by a huge margin and it will be hugely biased as non-fradulant so undersampling is a much better approach to get an optimal and desired outcome.

## Confusion Matrix
We create a user defined function for the confusion matrix or we can use `confusion_matrix` from `sklearn.matrics` library.

# Applying SVM
We train our mode by importing `svm` from `sklearn`. We used **'linear'** kernel (a more about kernel later in this project) to train our data for now, but we will change kernel afterwords.
The Syntax is as follows:
```
from sklearn import svm
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)
prediction_SVM = classifier.predict(X_train)
```

We get accuracy of our training model more than 95% most of the time with random samples.
The confusion matrix is as follows: 
<p align = center><img src = https://user-images.githubusercontent.com/54438860/120145390-0b94f780-c201-11eb-8976-8ed3476434b2.png></p>
<h5 align = center>Fig 3: Confusion matrix of training model</h5>

## Testing our model
To test our model, the syntax is as follows:
```
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train,y_train)
prediction_SVM_all = classifier.predict(X_test_all)
cm = confusion_matrix(y_test_all, prediction_SVM_all)
plot_confusion_matrix(cm,class_names)
```

The confusion matrix obtained is as follows:
<p align = center><img src = https://user-images.githubusercontent.com/54438860/120145760-adb4df80-c201-11eb-96bb-082f6ed905f2.png></p>
<h5 align = center>Fig 4: Confusion matrix of testing model</h5>

## Rebalancing the Class Weights
We need to minimize the False positives i.e, the number of non detected frauds to improve the performance of our model. We can do this by modifying the class_weight parameter, we can chose which class to give more importance during the training phase.

Syntax is as follows:
```
classifier_b = svm.SVC(kernel='linear',class_weight={0:0.60, 1:0.40})
classifier_b.fit(X_train, y_train)      # Then we train our model, with our balanced data train.
```
### Re-Testing the model
After re-testing we get our confusin matrix as follows:
<p align = center><img src = https://user-images.githubusercontent.com/54438860/120146385-9de9cb00-c202-11eb-9810-2dbcef0ca103.png></p>
<h5 align = center>Fig 5: Confusion matrix of testing model after rebalancing the calss weights</h5>

## Changing the Kernel
The SVM basically works in different kernels which are designed for different type of data distribution. By data distribution I mean how the data points are scattered along the hyperplane.

In other words, one can say that different kernels enables the SVM model to use different type of hyperplane on the dataset.

Thus, some of them are used below and the kernel which results in the minium error in the confusion matrix will be the bebst suited SVM kernel on the dataset. Hence, enabling the SVM algorithm to put it's best performance on the dataset.

For this project, we used four of the most used kernel; Namely **'Linear'**, **'Polynomial'**,**'Sigmoid'** and **'Radial basis function _(RBF)_'** kernel.
Above we saw the 'Linear' kernel.

### Polynomial Kernel
For polynomial kernel syntax is as follows:
```
classifier_b = svm.SVC(kernel='poly',class_weight={0:0.60, 1:0.40})
classifier_b.fit(X_train, y_train)
prediction_SVM_b_all = classifier_b.predict(X_test_all)
cm = confusion_matrix(y_test_all, prediction_SVM_b_all)
plot_confusion_matrix(cm,class_names)
```
The accuracy of our model is `99.79%` which is a lot more better when compared to the linear model's accuracy `95.94%`.
The confusion matrix is as follows:
<p align = center><img src = https://user-images.githubusercontent.com/54438860/120147477-56fcd500-c204-11eb-8929-27017ceb4cfb.png></p>
<h5 align = center>Fig 5: Confusion matrix of testing model by using Polynomial kernel</h5>

### Radial Basis Function (RBF) Kernel
For RBF kernel the syntax is as follows:
```
classifier_b = svm.SVC(kernel='rbf',class_weight={0:0.60, 1:0.40})
classifier_b.fit(X_train, y_train)
prediction_SVM_b_all = classifier_b.predict(X_test_all)
cm = confusion_matrix(y_test_all, prediction_SVM_b_all)
plot_confusion_matrix(cm,class_names)
```
After using RBF as a kernel, we got an accuracy of `97.38%` which is still better than the linear but not as good as polynomial.
It's Confusion matrix is as follows:
<p align = center><img src = https://user-images.githubusercontent.com/54438860/120148022-397c3b00-c205-11eb-937e-2d9b65e283ff.png></p>
<h5 align = center>Fig 6: Confusion matrix of testing model by using RBF kernel</h5>

### Sigmoid Kernel
For Sigmoid kernel the syntax is as follows:
```
classifier_b = svm.SVC(kernel='sigmoid',class_weight={0:0.60, 1:0.40})
classifier_b.fit(X_train, y_train)
prediction_SVM_b_all = classifier_b.predict(X_test_all)
cm = confusion_matrix(y_test_all, prediction_SVM_b_all)
plot_confusion_matrix(cm,class_names)
```
After using Sigmoid as a kernel, we get accuracy of `66.75%` which is much worse than other kernel. This is because our data is highly non-linear and cannot be properly classified using sigmoid function.
It's confusion matrix is as follows:
<p align = center><img src = https://user-images.githubusercontent.com/54438860/120148371-b9a2a080-c205-11eb-8f81-8381e10caa62.png></p>
<h5 align = center>Fig 7: Confusion matrix of testing model by using Sigmoid kernel</h5>

## Precision, Recall, F1-Score, Mean Absolute Error, Mean Percentage Error and Mean Squared Error
We can find Precision, Recall, F1-Score, Mean Absolute Error, Mean Percentage Error and Mean Squared Error using the following synatx:
```
from sklearn.metrics import classification_report,mean_absolute_error,mean_squared_error

report= classification_report(y_test_all, prediction_SVM_b_all)
print(report)

mean_abs_error = mean_absolute_error(y_test_all,prediction_SVM_b_all)
mean_abs_percentage_error = np.mean(np.abs((y_test_all - prediction_SVM_b_all) // y_test_all))
mse= mean_squared_error(y_test_all,prediction_SVM_b_all)
print("Mean absolute error : {} \nMean Absolute Percentage error : {}\nMean Squared Error : {}".format(mean_abs_error,mean_abs_percentage_error,mse))
```
