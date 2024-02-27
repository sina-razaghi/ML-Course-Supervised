**Question1**: Write a program that receives a number of arbitrary two-dimensional data points from the user via the mouse, and then draws the least squares regression line for the data.

**Answer**: To start, we need to collect an arbitrary number of data points from the user's mouse location on the screen. Next, it's time to calculate the equation 𝑦 = 𝑤₀ + 𝑤₁𝑥, for which we need to find 𝑤 and then extract 𝑤₀ and 𝑤₁. Finally, the line is drawn based on the formula 𝑦 = 𝑤₀ + 𝑤₁𝑥 − 𝑤₁𝑥(𝑡−1).

Here is a more detailed explanation of each step:
Collecting data points: We can use the ginput function from the matplotlib.pyplot library to get the coordinates of the points from the user.
Calculating the coefficients of the regression line: We can use the lstsq function from the numpy library to calculate the coefficients of the regression line.
Drawing the regression line: We can use the plot function from the matplotlib.pyplot library to draw the regression line.

![image](https://github.com/sina-razaghi/ML-Course-Supervised/assets/47954697/0050d0cd-8f3f-474b-bfee-eeb2ebe1b321)
![image](https://github.com/sina-razaghi/ML-Course-Supervised/assets/47954697/281bd8e5-8934-4492-804d-839bb7e1ae20)


**Question2**: Find a dataset related to a specific field in a way that is related to the regression problem. Then write a program to investigate the following items with the least square error regressor on this data:
A. Dimensions of the problem (number of features)
B. Number of samples
C. Noise level
D. Linearity and non-linearity of the relationship between input and output
E. Effect of regulation
F. A combination of the above
Then, for each of the above cases, after dividing the data of each set into training and experimental data (with the 10-fold method), for each data set separately, a linear regressor based on the least square error is calculated and the amount of error report with different criteria. Finally, interpret and analyze all your findings.

**Answer**

1. Choose a Dataset: **‫‪Prediction‬‬‫‪Speed‬‬ ‫‪Wind‬‬**
Start by selecting a dataset from a field of your interest that can be used for regression analysis. Here are some examples:
Field: Housing prices
Dataset: California housing prices (available on https://www.kaggle.com/datasets/mohamedbakrey/housecsv)
Field: Medical research
Dataset: Body mass index (BMI) and life expectancy (available on https://worldpopulationreview.com/)
I choose **‫‪Prediction‬‬‫‪Speed‬‬ ‫‪Wind‬‬**
![image](https://github.com/sina-razaghi/ML-Course-Supervised/assets/47954697/78550db9-412b-4b68-877c-edcade32b130)

3. Preprocessing:
Check for missing values: Identify any missing data points and consider appropriate handling methods like imputation or removal.
Feature scaling: If the features have different scales, consider scaling them using techniques like standardization or normalization to improve regression performance.

4. Investigation:
A. Dimensions (Number of Features):
Analyze the provided data description or documentation to understand the number of features.
Visualize the data: Explore the distribution of each feature using techniques like histograms or box plots. Look for any outliers or irregularities.

B. Number of Samples:
Check the size of the dataset.
Consider the trade-off between overfitting and underfitting: Having too few samples can lead to overfitting, while too many might increase computational cost.

C. Noise Level:
Calculate the signal-to-noise ratio (SNR): This ratio compares the variance of the signal (dependent variable) to the variance of the noise. A higher SNR indicates less noise.
Visualize the data: Look for any random fluctuations in the scatter plot of the data that might indicate noise.

D. Linearity vs. Non-linearity:
Visualize the relationship between the features and the target variable using scatter plots. Look for a linear or non-linear trend.
Calculate the correlation coefficient: A value close to +1 or -1 indicates a strong linear relationship, while values closer to 0 suggest a weaker or non-linear relationship.

E. Effect of Regularization:
Split the data into training and testing sets using a technique like k-fold cross-validation (e.g., 10-fold).
Train linear regressors with different regularization parameters (e.g., L1 or L2).
Evaluate the performance of each model on the testing set using metrics like mean squared error (MSE) or R-squared. Observe how regularization affects the model complexity and generalization.
![image](https://github.com/sina-razaghi/ML-Course-Supervised/assets/47954697/1edbbc4a-c1bb-46b4-9c60-df35cdcdfcc7)
![image](https://github.com/sina-razaghi/ML-Course-Supervised/assets/47954697/b3a70079-809c-4075-9530-62f62d679723)

F. Combined Analysis:
Combine the insights from your investigations in A-E to understand the overall properties of the data and how they might affect the performance of the regressor.

4. Reporting and Interpretation:
Summarize your findings for each of the investigated aspects.
Interpret the results: How do the findings relate to the chosen dataset and the field of study?
Draw conclusions: Based on your analysis, can a linear least square regressor be an appropriate choice for this data? Explain your reasoning.
![image](https://github.com/sina-razaghi/ML-Course-Supervised/assets/47954697/6a8cba69-a8f3-4556-a2ed-bf92010c855c)

