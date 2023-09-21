# ML Zoomcamp 2023
## Logistic Regression Analysis
*Author* : Javier Blanco

#### Disclaimer:
> Unlike linear regression, the normal equation cannot be used to estimate the parameters of the logistic regression model. This is because the sigmoid function is non-linear, and the negative log-likelihood function is not convex.
> 
*['SaturnCloud'](https://saturncloud.io/blog/can-we-use-normal-equation-for-logistic-regression/)*

### Introduction
While the above disclaimer text holds true, this article intends to demonstrate how we can adapt the Normal Equation used in Linear Regression, as taught in Module 2 of this course, for application in a Logistic Regression model. The objective here is to explore how the principles of the Normal Equation approach can be applied to this model, with careful considerations in mind:

1. When the value of "n" is small, such as when there are few features, the normal equation method performs effectively. This is because it involves the use of (X.T * X)^-1, which can be computationally costly.
2. With the normal equation approach, there's no requirement to select values for alpha or specify the number of iterations.

**Notice**: The commonly used approach in Logistic Regression for estimating regressor values (weights) is Maximum Likelihood Estimation (MLE). MLE is a method designed to identify the parameters that maximize the likelihood of observing the training data. Several iterative algorithms, such as *gradient descent*, are available for parameter estimation. These algorithms are primarily focused on minimizing the negative log-likelihood function. 

### Logistic Regression - Fundaments
Logistic Regression (LR) serves as a method applied in binary classification scenarios, where the outcome variable assumes one of two values, typically 0 or 1. Unlike linear regression, logistic regression does not produce a continuous outcome; instead, it produces a probability that ranges between 0 and 1.

> *Logistic Regression as a special case of Generalized Linear Models (GLM)* with a Binomial/Bernoulli conditional distribution and a Logit link 
> [scikit-learn.org](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) 
 

To determine this probability value, the logistic regression model uses a sigmoid function as follows:

$$                                                                                                                                                                                                                                                                                       \sigma                                                                                                                                                                                                                                                                                       ( z ) = \frac                                                                                                                                                                                                                                                                                       { 1 }                                                                                                                                                                                                                                                                                       { 1 + e^ { -z }                                                                                                                                                                                                                                                                                     }                                                                                                                                                                                                                                                                                       $$

where `z` acts as the linear combination of the input characteristics and parameters. 

Once the sigma curve is fitted, LR predicts the probability of the positive class $P ( y_i=1|X_i )$ using the following expression:

$$  \hat  { p }  ( X_i )    = \frac  { 1 }  { 1 + \exp  ( -X_i w - w_0 )  }  $$

where: 

$z = X_i w + w_0$
$w, w_0:$ weights (coefficients) of the LR model

The following image shows the graphical representation of the sigmoid function built in [Geogebra Suite](https://www.geogebra.org/calculator):

![LR](https://drive.google.com/uc?export=view&id=1TGmh7c6t6MvYPFvYRze_ys7CEryCNujX)

Remember that LR is a continuous function with a sigmoid shape. It estimates the probability of the target value being either above or below a threshold, typically set at 0.5. In the former case, the target value is classified as 1; otherwise, it's classified as 0. This is why LR is considered a classification model. 

* **Notice**: *LR can also be used for multinomial classification, where the target variable has multiple classes. However, this article does not cover such cases.*

### Normal Equation approach
Once we have understood the mathematical expression that underlies LR, we will apply the concepts of the normal equation, as covered in the Linear Regression class in Module 2 of the [Machine Learning course](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/02-regression). Both the weights and the parameters of the expression are integral components of `z`.
  
#### Model expression
Normal Equation representation involves an input variable (features) tipically represented as `X`, and an output variable (target), represented as `y`. This target variable has two classes, usually between $[0,1]$ domain. 
We also need to understand the dataset's structure, where `n` represents the number of rows, which are the training examples, and `m` denotes the number of features. It includes all columns in the dataset except for the target column. 

To present this more formally:

$X:$ input variable (features or columns with not target)
$y:$ output variable (target column, with two posible classes)
$n:$ number of examples (rows)
$m:$ number of columns or features to use
$X(i):$ that is i-th example of training data
$y(i):$ i-th example of target data. 

The matrix representation of a classification problem is presented below:

* **Feature matrix or X(Matrix)**
$$ X = \begin{bmatrix}
w_{0} & X_{1} & X_{2} & X_{3} & X_{n} \\ 
1 & a_{12} & a_{13} & ... &a_{1n} \\ 
1 & a_{21} & a_{22} & ... & a_{2n} \\
1 & a_{31} & a_{32} & ... & a_{3n} \\
... & ...      & ...      & ... & ....  \\
1 & a_{n1} & a_{n2} & a_{n3} & a_{nm}  
\end {bmatrix}_{(m,n+1)}$$

*  **Target Vector**
$$ Y = \begin{bmatrix} 
1 \\ 
0  \\
1  \\
...  \\
1 
\end {bmatrix}_{(m,1)}$$

* **Weight Matrix (also known as $\Theta$ (Theta) matrix**

$$ W = \begin{bmatrix} 
w_0 \\ 
w_1  \\
w_2  \\
...  \\
w_n 
\end {bmatrix}_{(n+1, 1)}$$

The initial value of the weight matrix, `W`, is set to zero. To estimate the predicted `y` with minimum error compared to the actual value of `y`, we need to calculate the model's weights (coefficients). In this article, we will refer to the predicted y as `g(z)` or `y_pred`  

#### Vectorization
In the preceding section, we discussed the sigmoid function, which takes the following arguments::
$$z = X_i w + w_0$$
So, to estimate the predicted target values, we perform a matrix multiplication between `W` and `X`. As a result, the predicted `y` can be represented as follows::

$$y_{pred} = g(z)$$
whereas,
$$y_{pred} = g(w_{0}*X0_i + w_{1}*X1_i )$$

Here, `g(z)` is a sigmoid function that yields values between 0 and 1, as mentioned previously.

if $y_{pred} >= 0.5$ then `g(z)` = 1
if $y_{pred} < 0.5$ then `g(z)` = 0

Expanding on the matrix approach, we have:
$$Z = X*W$$
Now, making a change of variables:

$$g \left (\begin{bmatrix} 
1 & a_{12} & a_{13} & ... &a_{1n} \\ 
1 & a_{21} & a_{22} & ... & a_{2n} \\
1 & a_{31} & a_{32} & ... & a_{3n} \\
... & ...      & ...      & ... & ....  \\
1 & a_{n1} & a_{n2} & a_{n3} & a_{nm}  
\end {bmatrix} * \begin{bmatrix} 
w_0 \\ 
w_1  \\
w_2  \\
...  \\
w_n 
\end {bmatrix}\right) = \begin{bmatrix} 
1 \\ 
0  \\
1  \\
...  \\
1 
\end {bmatrix} $$

Or, equivalently:

$$ y_{pred}    = \frac  { 1 }  { 1 + \exp  ( -X*W )  } $$

To find the value of W we are going to use Normal Equation approach:

$$W = (X^{T} * X)^{-1} * X^{T} * y^T$$

where:

$X^{T}$ is transposed of X
$y^{T}$ is transposed of y

We also know that $Z = X*W$, therefore:

$$Z = X*[ (X^{T} * X)^{-1} * X^{T} * y^T]$$
$$y_{pred} = \frac  { 1 }  { 1 + \exp^{-(X*W )}  } $$
whereas
$$y_{pred} = \frac  { 1 }  { 1 + \exp ^{-( X*[ (X^{T} * X)^{-1} * X^{T} * y^T] ) } }$$

#### Cost Function
To assess the accuracy of our predictions compared to validation target values in binary classification tasks, the Log-loss metric is commonly used as a cost function. Here, we present a generic formula for the Log-loss metric.:

$$-\frac 1N \sum_{i=1}^{N}y_i * log(p(y_i)) + (1 - y_i) * log(1-p(y_i))$$

where:
$p(y_i):$ probability of 1.
$1-p(y_i):$ probability of 0.

To calculate the Log-loss, the formula involves three important steps:
1. Find corrected probabilities
2. Calculate the logarithm of corrected probabilities
3. Take negative average of the values obtained in the second step

As you can see, Log-loss function provides a continuos measure of model's performance, being a suitable option as optimization algorithm. 
According ScikitLearn, binary class logistic regression minimizes the Log-loss function. 

> To delve deeper into the Log-loss cost function, you can refer to this [link](https://www.internalpointers.com/post/cost-function-logistic-regression) which is an accurate resource about this topic. 

In our example, we will simplify the Log-loss formula.:

$$J(z) =-\frac 1m   * (y ^T*log(g(z)(X (i))) +(1 − y )^T*log(1 − g(z)(X(i))))$$

Where:

$m:$ Number of examples in train data (rows)
$T:$ Transposed matrix
$g(z):$ y predicted 

### Practice Example
For this example we are going to use [HCV dataset](https://archive.ics.uci.edu/dataset/571/hcv+data) obtained from UCI repository. 
The data set contains laboratory values of blood donors and Hepatitis C patients along with demographic values like age. The target attribute for classification is Category:
{'0=Blood Donor', '0=suspect Blood Donor', '1=Hepatitis', '2=Fibrosis', '3=Cirrhosis'}

*You can see full code notebook following this [link]()*

```python
    #Import libraries
    import pandas as pd
    import numpy as np
```
After uploading data from UCI repository we get a table like this:

|index|Category|Age|Sex|ALB|ALP|ALT|AST|BIL|CHE|CHOL|CREA|GGT|PROT|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|0=Blood Donor|32|m|38\.5|52\.5|7\.7|22\.1|7\.5|6\.93|3\.23|106\.0|12\.1|69\.0|
|2|0=Blood Donor|32|m|38\.5|70\.3|18\.0|24\.7|3\.9|11\.17|4\.8|74\.0|15\.6|76\.5|
|3|0=Blood Donor|32|m|46\.9|74\.7|36\.2|52\.6|6\.1|8\.84|5\.2|86\.0|33\.2|79\.3|
|4|0=Blood Donor|32|m|43\.2|52\.0|30\.6|22\.6|18\.9|7\.33|4\.74|80\.0|33\.8|75\.7|
|5|0=Blood Donor|32|m|39\.2|74\.1|32\.6|24\.8|9\.6|9\.15|4\.32|76\.0|29\.9|68\.7|

Since this post does not cover multinomial classification, we will need to transform the multiclass variable (`Category`) into a binary class (0,1).
Hence, `Blodd Donor` and `suspect Blood Donor` are going to set as `1`, and the other categories as `0` (`Hepatitis`, `Fibrosis`, `Cirrhosis`).

It's also important to determine the count of instances set as `1` and  `0`, so we will create a bar plot to visualize this as follows:

```python
hcv_data['Category'] = [(1  if category == '0=Blood Donor'  or category =='0=suspect Blood Donor'  else  0)  for category in hcv_data['Category']]
count_cat = hcv_data['Category'].value_counts().reset_index()
count_cat.columns = ['Category',  'Count']
count_cat.plot(kind='bar', x='Category', y='Count', legend=False)
```
    

![](data:image/png;base64,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9evXT6+99lrYuM1m09KlS3XHHXcoMTFRTz/9tPr06XPWR6rv2rVLNptNBw4caL0vIIBzZwFAM7788kvLZrNZzz777HfOW7RokbVp0ybr0KFDVklJidWvXz9r0qRJlmVZVjAYtF544QXL4XBYR44csY4cOWIdP37csizLevjhh60bbrjB2rJli3XgwAFrwYIFlt1ut/71r39ZlmVZH3zwgRUTE2MtWLDAKi8vt5YsWWIlJydbTqcz9NqrV6+24uPjrSVLlljl5eXW888/b8XGxlqbNm0KzZFkpaamWsuWLbMOHjxoff7559Yzzzxjpaenh53Hr371K2vYsGGt8aUD0AoIFADN2r59uyXJWr16dUTHrVq1ykpJSQk9Xr58eVhUWJZlff7551ZsbKz1xRdfhO0fPny4lZ+fb1mWZY0ZM8bKyckJGx87dmzYc91www3WxIkTw+bcc8891q233hp6LMmaOnVq2JwvvvjCio2NtbZv325ZlmU1NDRYXbp0sYqLiyM6VwDnD7/iAdAs6wf+FYx3331Xw4cPV48ePdS5c2eNGzdOX375pb7++utvPWb37t1qampS37591alTp9C2efNmHTx4UJJUXl6u66+/Puy4bz7eu3evhg4dGrZv6NCh2rt3b9i+wYMHhz3u3r27cnJytGzZMknSmjVrFAwGdc899/ygcwZw/nGTLIBmXXHFFbLZbN95I+zhw4d12223adKkSXrmmWeUnJysDz74QBMmTFBDQ4M6duzY7HEnTpxQbGysysrKFBsbGzbWqVOnVj0PSUpMTDxr38MPP6xx48Zp0aJFWr58ucaMGfOt6wVw4XEFBUCzkpOTlZ2drSVLlqi+vv6s8draWpWVlen06dN6/vnnlZmZqb59+6qqqipsXkJCgpqamsL2XXvttWpqalJNTY369OkTtrndbklSv379tGPHjrDjvvl4wIAB+vDDD8P2ffjhh0pPT//e87v11luVmJiopUuXasOGDXrooYe+9xgAFw6BAuBbLVmyRE1NTbr++uv1l7/8Rfv379fevXu1ePFieTwe9enTR42NjXrxxRf12Wef6bXXXlNRUVHYc/Tu3VsnTpxQSUmJjh07pq+//lp9+/bV2LFjdf/992v16tU6dOiQSktLVVBQoHXr1kmSpkyZor///e9auHCh9u/fr5deeknr16+XzWYLPfeMGTNUXFyspUuXav/+/Vq4cKFWr16t6dOnf++5xcbG6oEHHlB+fr6uuOIKeTye1v3iAWiZaN8EA8BsVVVVltfrtXr16mUlJCRYPXr0sO644w7rH//4h2VZlrVw4UKrW7duVocOHazs7Gzr1VdftSRZ//nPf0LP8eijj1opKSmWJGv27NmWZf33xtRZs2ZZvXv3tuLj461u3bpZd911l/XJJ5+Ejvv9739v9ejRw+rQoYM1atQo6+mnn7bcbnfY+goLC63LL7/cio+Pt/r27Wu9+uqrYeOSrDfffLPZczt48KAlyZo/f36Lv04AWpfNsn7gnXAAEGUTJ07Uvn379P7777fK873//vsaPny4Kisr5XK5WuU5AbQObpIFYKznnntON998sxITE7V+/XqtWLFChYWFLX7eYDCoo0ePas6cObrnnnuIE8BA3IMCwFilpaW6+eabNXDgQBUVFWnx4sV6+OGHW/y8f/zjH9WrVy/V1tZq/vz5rbBSAK2NX/EAAADjcAUFAAAYh0ABAADGIVAAAIBxCBQAAGAcAgUAABiHQAEAAMYhUAAAgHEIFAAAYJz/Ax9O5dkybACAAAAAAElFTkSuQmCC)

As you can observe, we have an imbalanced class situation. In such cases, resampling methods like oversampling or undersampling are typically employed to address this issue. However, please note that we won't be covering these methods in this article.

As a result of applying data transformation we have:

|index|Category|Age|Sex|ALB|ALP|ALT|AST|BIL|CHE|CHOL|CREA|GGT|PROT|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|1|32|m|38\.5|52\.5|7\.7|22\.1|7\.5|6\.93|3\.23|106\.0|12\.1|69\.0|
|2|1|32|m|38\.5|70\.3|18\.0|24\.7|3\.9|11\.17|4\.8|74\.0|15\.6|76\.5|
|3|1|32|m|46\.9|74\.7|36\.2|52\.6|6\.1|8\.84|5\.2|86\.0|33\.2|79\.3|
|4|1|32|m|43\.2|52\.0|30\.6|22\.6|18\.9|7\.33|4\.74|80\.0|33\.8|75\.7|
|5|1|32|m|39\.2|74\.1|32\.6|24\.8|9\.6|9\.15|4\.32|76\.0|29\.9|68\.7|

#### Attributes explanation:

Following the variables used in the [Hoffmann et al.(2018)](https://jlpm.amegroups.org/article/view/4401/5425) paper, we will consider:

* `Category`: 1= Blood Donor or suspect of Blood Donor (Categorical)

* `ALB`: albumin levels (Continuous)

* `BIL`: bilirubin levels (Continuous)

* `CHE`: choline esterase levels (Continuous)

* `GGT`: γ-glutamyl-transferase levels (Continuous)

* `AST`: aspartate amino-transferase levels (Continuous)

* `ALT`: alanine amino-transferase levels


#### Feature selection an preparation
We will now select features from the dataset and verify if there are any missing values. If any are found, we will fill them with the mean values of their respective columns..
```python
hcv_df = hcv_df[['ALB',  'BIL',  'CHE',  'GGT',  'AST',  'ALT',  'Category']]
hcv_df.isna().sum()
hcv_df['ALB'] = hcv_df['ALB'].fillna(hcv_df['ALB'].mean())
hcv_df['ALT'] = hcv_df['ALT'].fillna(hcv_df['ALT'].mean())
```

#### Split dataset
The dataset needs to be divided into train, validation, and test subsets, following a proportion of 60%/20%/20%.

```python
def split_dataset(df, target_name, train_prop, val_prop, test_prop, random_state):
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    y = df[target_name]
    X = df.drop(columns=target_name,axis=1)
    train_indexes = int(len(df)*train_prop)
    val_indexes = int(len(df)*val_prop)
    X_train = X[:train_indexes]
    X_val = X[train_indexes:train_indexes + val_indexes]
    X_test = X[train_indexes + val_indexes:]
    y_train = y[:train_indexes]
    y_val = y[train_indexes:train_indexes + val_indexes]
    y_test = y[train_indexes + val_indexes:]
    data_splitted = [X_train, y_train, X_val, y_val, X_test, y_test]
    return data_splitted
```

#### Logistic Regression model:
X: Features
y: target

$$X = \begin{bmatrix}
w0 & ALB & BIL & CHE & GGT & AST & ALT \\
1 & 38.5 & 7.5 & 6.93 & 12.1 & 22.1 & 7.7 \\
1 & 38.5 & 3.9 & 11.17 & 15.6 & 24.7 & 18.0 \\
1 & 46.9 & 6.1 & 8.84 & 33.2 & 52.6 & 36.2 \\
1 & 43.2 & 18.9 & 7.33 & 33.8 & 22.6 & 30.6 \\
... & ... & ... & ... & ... & ... & ...
\end{bmatrix}$$

$$y = \begin{bmatrix}
category\\
1\\
1\\
1\\
1\\
...
\end{bmatrix}$$

$$W = \begin{bmatrix}
weights\\
0\\
0\\
0\\
0\\
...
\end{bmatrix}$$

 
First, we'll add $W_0$ values to the feature matrix in our dataset. Then, we'll train the model using the Normal Equation approach and validate its performance with the validation data using the cost function formula. Let's define the necessary functions.
```python
X_train_stacked = pd.concat([pd.Series(1, index=X_train.index, name='W0'), X_train], axis=1)
X_train_stacked.head()
```
Feature matrix looks like this: 

|index|W0|ALB|BIL|CHE|GGT|AST|ALT|
|---|---|---|---|---|---|---|---|
|0|1|28\.1|2\.8|5\.58|26\.2|17\.5|16\.6|
|1|1|31\.4|2\.4|5\.95|22\.9|17\.0|16\.6|
|2|1|43\.7|8\.1|8\.15|13\.4|26\.3|17\.3|
|3|1|32\.0|50\.0|5\.57|650\.9|110\.3|5\.9|
|4|1|35\.5|6\.4|8\.81|24\.1|29\.5|27\.5|

and now, we can calculate $W$ matrix values using the following formula:
$$W = (X^{T}  * X)^{-1} * X^{T} * y^T$$

In python:
```python
X = np.matrix(X_train_stacked)
y = np.matrix(y_train).T
W = np.linalg.inv(X.T*X) * X.T * y
```
As a result, we have:

$$W = \begin{bmatrix} 
8.58051212e-01 \\
6.97079801e-03 \\
-2.24239344e-03 \\
-1.55449883e-03 \\
-1.40208916e-03 \\
-6.10794657e-03 \\
2.60820872e-04 \\
\end{bmatrix}$$

Let's proceed to calcualte sigmoid values:
```python
y_pred = 1 / (1 + np.exp(-(X * W)))
```

Also, we need to check model performance using Log-loss formula:

```python
#cost function
m = len(y)
J = (-1/m)*(y.T*np.log(y_pred) + (1 - y).T * np.log(1 - y_pred))
J = J[0,0]
J
```
cost value `42,39%`

Let's validate performance model with validation data:
```python
 def logistic_regression(X, W, y):
   	X = pd.concat([pd.Series(1, index=X.index, name='W0'), X], axis=1)
   	X = np.matrix(X)
   	y = np.matrix(y).T
   	y_pred = 1 / (1 + np.exp(-(X * W)))
   	m = len(y)
   	J = (-1/m)*(y.T*np.log(y_pred) + (1 - y).T * np.log(1 - y_pred))
   	J = J[0,0]
   	return y_pred, J

y_pred, cost_value = logistic_regression(X=X_val, W=W, y=y_val)
cost_value
```
cost value: 40.02%

Now, let's combine train and validation sets to test the model:
```python
X_all = pd.concat([X_train, X_val, X_test], axis=0)
y_all = pd.concat([y_train, y_val, y_test], axis=0)

y_pred, cost_value = logistic_regression(X=X_all, W=W, y=y_all)
cost_value
```
Cost value: 41.29%

Finally, we need to find the accuracy of the model:
```python
y_test_pred, cost_value = logistic_regression(X=X_test, W=W, y=y_test)
y_test_pred = np.array([1  if value >= 0.5  else  0  for value in y_test_pred])
y_test = np.array(y_test)
k = np.double(y_test_pred == np.array(y_test))
acc = np.mean(k)*100
acc
```
Accuracy of 91.05%

## Conclusion
After exploring the adaptation of the Normal Equation approach, typically used in Linear Regression, for Logistic Regression, we delved into the practical application of Logistic Regression, which relies on a sigmoid function to estimate probabilities in binary classification. Our exploration revealed that the Normal Equation can be effectively employed to determine the model's weights, enabling the calculation of predicted probabilities without resorting to iterative methods. It's worth noting, however, that this approach is most effective when dealing with a limited number of features and in cases where parameters like alpha or iteration counts are not required.

Furthermore, we introduced the Log-loss cost function as a reliable means to assess model accuracy, and we provided a hands-on example using an HCV dataset for practical application. In summary, our discussion underscores the viability of employing the Normal Equation in Logistic Regression, especially in scenarios characterized by a small number of features, yielding results that are notably accurate and applicable.

## References

* Dhameliya, P. (2020). Logistic Regression — Implementation from scratch. https://medium.com/@pdhameliya3333/logistic-regression-implementation-from-scratch-3dab8cf134a8
* Logistic Regression. https://saturncloud.io/glossary/logistic-regression/
* Scikit Learn Documentation. https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
* Sethia M, (2023). Binary Cross Entropy aka Log Loss-The cost function used in Logistic Regression. https://www.analyticsvidhya.com/blog/2020/11/binary-cross-entropy-aka-log-loss-the-cost-function-used-in-logistic-regression/
* Mazumder S. (2022). 5 Techniques to Handle Imbalanced Data For a Classification Problem. https://www.analyticsvidhya.com/blog/2021/06/5-techniques-to-handle-imbalanced-data-for-a-classification-problem/
