# ML Zoomcamp 2023
## Logistic Regression Analysis 
### (Part II)
*Author* : Javier Blanco

In our [previous post](https://github.com/jblanco89/ML-Zoomcamp-course/blob/main/Logistic_Regression_analysis.md), we explored the practicality of applying the Normal Equation (NE) approach to a Logistic Regression (LR) model, taking into account specific constraints such as a small dataset, limited features, and binary classification targets. This choice was driven by the computational cost associated with the NE approach. Furthermore, we delved into the application of the Log-loss equation as our chosen cost function and provided a detailed explanation of its usage.

To assess the model's performance, we relied on the accuracy metric, which yielded an impressive accuracy rate of 91%. Additionally, in a post-hoc analysis, we calculated an F-1 score of 95% on the tested [dataset](https://archive.ics.uci.edu/dataset/571/hcv+data), showcasing the effectiveness of the NE approach.

However, it's worth noting that this was primarily a practical exercise aimed at diving deeply into Linear Algebra concepts covered in Module 2 of the [ML Zoomcamp course](https://github.com/DataTalksClub/machine-learning-zoomcamp). Several questions arose during this exercise. For example, how do we determine if the estimated coefficients (weights) are the best fit for the model? What about other metrics typically used in classification techniques, such as the Confusion Matrix or F-1 Score? Additionally, which optimization methods are available to enhance model performance? In this post, we will explore some of these questions.   

### Non-Constrained optimization techniques

Also known as **Convex Functions of Optimization**, they are methods to find the optimal solution to a problems where we need to know input values that minimize or maximize an convex objective function. Whithin convex functions we can highlight the two most used:
1. Gradient Descent
2. Newton's Method

Gradient Descent is a method for optimizing a model's parameters by adjusting them according to the gradient of the objective function. It starts with an initial parameter set and iteratively moves in the direction opposite to the gradient to minimize the objective function, basically you have to iterate over the training dataset while re-adjusting the model. This approach is versatile and can be used with both convex and non-convex functions. **The goal is to minimize the cost function because it means you get the smallest possible error and improve the accuracy of the model.**

> *There are different types of optimization algorithms, including those for convex, non-convex, and constrained functions.*
> [Brownlee, J. (2020)](https://machinelearningmastery.com/tour-of-optimization-algorithms/)

For this post's resolution, we'll utilize Gradient Descent (GD) as the optimization function to discover the optimal values of `W` that minimize the Log-loss function (cost function) and enhance the model's prediction performance. I plan to delve into Newton's method in a future post.

### Practice Application
Up to this point, we already comprehended prediction function (`sigma(z)`), and we have utilized the Log-loss metric to assess how well our predictions align with the data. However, it would be interesting to determine if the values of `W` estimated with the NE are the most accurate we can obtain. This is where optimization techinques as Gradient Descent (GD) comes into play.

Staying within the nomenclature used in this post, we can represent the `GD` equation as follows:

$$W_{n+1} = W_{n} - \alpha * P_n$$

Where $P_n$ is the gradient or descent direction and $\alpha$ is the step size (or learning rate).

In addition, we may define the gradient as:

$$P_n = \frac {\partial J(z)}  {\partial z}$$

By doing partial derivative of cost function `J` we have:

$$P_n = \frac 1m * ((y_{pred} - y) * (X))^T$$
for more information about partial derivative see this [medium post](https://stackedit.io/app#providerId=googleDriveWorkspace&folderId=1nPkvxKlg0ByepLZ56hkvEi36BEOWtm-a) which describe the process step by step. 


`GD` technique consists in repeat $W_{n+1} = W_{n} - \alpha * P_n$ until cost function `J(z)` becomes minimum.


$\alpha$ (learning rate) is a hyper-parameter of the `GD` and this one should not be too small or too large. Learning rate commonly used are: `[0.001,0.01,0.1,1]`
To see the decreasing of cost function we simply need to plot $J(z)$ with the number of iterations we already set.

```python
#gradient function

def  gradient_descent(X, y_pred, y):
	d_Z = (1/m) * (X.T * (y_pred - y))
	return d_Z
```
now, Linear regression function should be:

```python
def  logistic_regression(X, W, y, alpha, epoch):
	X = pd.concat([pd.Series(1, index=X.index, name='W0'), X], axis=1)
	X = np.matrix(X)
	y = np.matrix(y).T
	m = len(y)
	history = []
	for i in  range(epoch):
		y_pred = 1 / (1 + np.exp(-(X * W)))
		J = (-1/m)*(y.T*np.log(y_pred) + (1 - y).T * np.log(1 - y_pred))
		grad = gradient_descent(X, y_pred, y)
		W = W - alpha*grad
		history = np.append(history, J)
		print(f'Epoch: {i} -- Cost: {J}')
		
	if epoch > 5:
	x = np.linspace(0,epoch,epoch)
	plt.ylabel("cost function")
	plt.plot(x,history,color='b')
	plt.xlabel("N iterations")
	plt.title("Cost function profile")
	plt.show()
return W, y_pred
```
now, we can try the function in this way:

```python
W_opt, y_pred_opt = logistic_regression(X=X_all, 
										W=W, 
										y=y_all, 
										alpha=0.001, 
										epoch=30)
```
Optimized weigths are now:
$$W = \begin{bmatrix} 
0.857 \\
0.080 \\
-0.014 \\
0.013 \\
-0.016 \\
-0.043 \\
-0.008 \\
 \end{bmatrix}$$

We can visualize cost function decreasing here:


![cost-decreasing-result](https://drive.google.com/uc?export=view&id=1LSuy_C0qJk8sC6pPe3yNgJQZj8Oqo_Ho)

As you can see, mantaining a learning rate of ($\alpha = 0.001$) constant and with `30 epochs`, The cost function decreased from `0.41` to `0.18`, indicating an improvement of approximately `43%` in model weight estimation. 

#### Model Evaluation 
To measure model performance, accuracy, f-1 score and Confusion Matrix have been used. To calculate them, we have used `metrics` class of [Scikit-Learn library](https://scikit-learn.org/stable/modules/model_evaluation.html)  
```python
def  eval_metrics(y, y_pred, metrics =['acc',  'f1',  'cm']):
	results = {}
	if  'acc'  in metrics:
		y_pred_binary = np.array([1  if value >= 0.5  else  0  for value in y_pred])
		acc = accuracy_score(y, y_pred_binary)
		results['acc'] = acc
	if  'f1'  in metrics:
		f1 = f1_score(y, y_pred_binary, zero_division='warn')
		results['f1'] = f1
	if  'precision'  in metrics:
		pre = precision_score(y, y_pred_binary)
		results['precision'] = pre
	if  'recall'  in metrics:
		rec = recall_score(y, y_pred_binary)
		results['recall'] = rec
	if  'cm'  in metrics:
		cm = confusion_matrix(y, y_pred_binary)
		results['cm'] = cm
	if  'report'  in metrics:
		report = classification_report(y, y_pred_binary)
		results['report'] = report
	return results
``` 
The combined training and validation datasets for features and targets have been labeled as `X_all` and `y_all`, respectively. With the `GD` optimization, the training and validation data achieved an accuracy of 93% and an f-1 score of 96.4%, compared to 91% accuracy and 95% f-1 score without `GD` optimization. In testing, we obtained an f-1 score of 98% and an accuracy of 96%.

![Confusion-Matrix](https://drive.google.com/uc?export=view&id=1AXFD2H-ZICxqfIhFPNjbreWkxMxUOnt4)

### Conclusion
We have employed the gradient descent optimization technique to enhance the prediction performance of our logistic regression model. In comparison to the Normal Equation approach, utilizing an optimization technique enables us to iteratively estimate more suitable coefficients (weights) by following the gradient direction of the objective function (Log-loss). It's important to highlight that keeping the learning rate constant is advisable, and it should neither be too small nor too large.
Furthermore, it's worth highlighting that there are several other optimization techniques available as it was mentioned. These methods are capable of yielding even better results, especially when dealing with large datasets containing thousands of features. One such technique is the Quasi-Newtonian method, and I plan to dive deeper into these approaches in a future post. 


