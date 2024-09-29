# Building my first linear regression model
# Simple Linear Regression 


```python
#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sklearn 
import seaborn as sns
```


```python
#load dataset
penguins = sns.load_dataset("penguins")

penguins.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>species</th>
      <th>island</th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195.0</td>
      <td>3250.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>193.0</td>
      <td>3450.0</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
</div>



From the first 5 rows, we can see the the columns names available: 
There is also some missing data.

# data cleaning


For the purposes of this project, we are focusing our analysis on Adelie and Gentoo penguins,
and will be dropping any missing values from the dataset. 


```python
#drop missing values
penguins.dropna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody atr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>species</th>
      <th>island</th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195.0</td>
      <td>3250.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>193.0</td>
      <td>3450.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.3</td>
      <td>20.6</td>
      <td>190.0</td>
      <td>3650.0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>338</th>
      <td>Gentoo</td>
      <td>Biscoe</td>
      <td>47.2</td>
      <td>13.7</td>
      <td>214.0</td>
      <td>4925.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>340</th>
      <td>Gentoo</td>
      <td>Biscoe</td>
      <td>46.8</td>
      <td>14.3</td>
      <td>215.0</td>
      <td>4850.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>341</th>
      <td>Gentoo</td>
      <td>Biscoe</td>
      <td>50.4</td>
      <td>15.7</td>
      <td>222.0</td>
      <td>5750.0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>342</th>
      <td>Gentoo</td>
      <td>Biscoe</td>
      <td>45.2</td>
      <td>14.8</td>
      <td>212.0</td>
      <td>5200.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>343</th>
      <td>Gentoo</td>
      <td>Biscoe</td>
      <td>49.9</td>
      <td>16.1</td>
      <td>213.0</td>
      <td>5400.0</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
<p>333 rows × 7 columns</p>
</div>




```python
#keep Adelie and gentoo penguins,drop missing values
penguins_sub = penguins[penguins["species"] != "Chinstrap"]
penguins_final = penguins_sub.dropna()
penguins_final.reset_index(inplace = True, drop = True)
```

# Exploratoy Data Analysis
before constructing any model I perfrom eda on the dataset to know more about the data.
we need to check for any linear relatiomship amoung variables inthe dataframe. 
library: searborn


```python
sns.pairplot(penguins_final)
```




    <seaborn.axisgrid.PairGrid at 0x1a40702e240>


![output_9_1](https://github.com/user-attachments/assets/5978ca79-368c-4391-9898-a72fbdd90dd4)




    


From the scatterplot matrix, we can observe a few linear relationships:
   - bill length(mm) and flipper length(mm)
   - bill length(mm) and body mass(g)
   - flipper length(mm) and body mass (g)

# Model Construction

Based on the above scatterplots, you could probably run a simple linear regression on any of the three relationships identified. For this part of the course,
you will focus on the relationship between bill length (mm) and body mass (g).


```python
# subset data
ols_data = penguins_final[["bill_length_mm", "body_mass_g"]]

```

next, we can construct the linear regressiob formula and save it
as a string. Remember that the y or dependent variable 
before the ~, and the x or independent variables comes after the ~. 


```python
# Write out formula
ols_formula = "body_mass_g ~ bill_length_mm"

```


```python
# import the ols fom the statsmodel
from statsmodels.formula.api import ols

```


```python
#build ols fit the model
OLS = ols(formula = ols_formula, data = ols_data)
model = OLS.fit()
```

Lastly, we can call the summary() function on the model object to get the coefficients and more statistics about the model. The output from model.summary() can be used to evaluate the model and interpret the results. Later in this section, we will go over how to read the results of the model output.


```python
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>body_mass_g</td>   <th>  R-squared:         </th> <td>   0.769</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.768</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   874.3</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 28 Sep 2024</td> <th>  Prob (F-statistic):</th> <td>1.33e-85</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:21:24</td>     <th>  Log-Likelihood:    </th> <td> -1965.8</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   265</td>      <th>  AIC:               </th> <td>   3936.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   263</td>      <th>  BIC:               </th> <td>   3943.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>      <td>-1707.2919</td> <td>  205.640</td> <td>   -8.302</td> <td> 0.000</td> <td>-2112.202</td> <td>-1302.382</td>
</tr>
<tr>
  <th>bill_length_mm</th> <td>  141.1904</td> <td>    4.775</td> <td>   29.569</td> <td> 0.000</td> <td>  131.788</td> <td>  150.592</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 2.060</td> <th>  Durbin-Watson:     </th> <td>   2.067</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.357</td> <th>  Jarque-Bera (JB):  </th> <td>   2.103</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.210</td> <th>  Prob(JB):          </th> <td>   0.349</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.882</td> <th>  Cond. No.          </th> <td>    357.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
#visualize the regression plot
sns.regplot(x = "bill_length_mm" , y = "body_mass_g", data = ols_data)
```




    <Axes: xlabel='bill_length_mm', ylabel='body_mass_g'>




    
![output_19_1](https://github.com/user-attachments/assets/f27b44a1-757d-4549-9a00-fe1a9f6b4a9e)

    


# Finish checking model assumptions
As you learned in previous videos, there are four main model assumptions for simple linear regression, in no particular order:

1. Linearity 
2. Normality
3. Independent observations  
5. Homoscedasticity.

we already checked the linearity assumption by creating the scatterplot matrix. The independent observations assumption is more about data collection. There is no reason to believe that one penguin's body mass or bill length would be related to any other penguin's anatomical measurements. So we can check off assumptions 1 and 3.

The normality and homoscedasticity assumptions focus on the distribution of errors. Thus, you can only check these assumptions after you have constructed the model. To check these assumptions, you will check the residuals, as an approximation of the errors.

To more easily check the model assumptions and create relevant visualizations, you can first subset the X variable by isolating just the bill_length_mm column. Additionally, we  can save the predicted values from the model using the model.predict(X) function.


```python
# subset X variable
X = ols_data["bill_length_mm"]

# get prediction from the model
fitted_values = model.predict(X)
```


```python
# residual 
residuals = model.resid
```


```python
#check normality assumption using the histplot to check normality
fig = sns.histplot(residuals)
fig.set_xlabel("Residual value")
fig.set_title("histogram of residuals")
plt.show

```




    <function matplotlib.pyplot.show(close=None, block=None)>




    
![png](output_23_1.png)
    


Another way to check the normality function is to create a quantile-quantile or Q-Q plot. Recall that if the residuals are normally distributed, you would expect a straight diagonal line going from the bottom left to the upper right of the Q-Q plot. You can create a Q-Q plot by using the qqplot function from the statsmodels.api package.

The Q-Q plot shows a similar pattern to the histogram, where the residuals are mostly normally distributed, except at the ends of the distribution.


```python
import statsmodels.api as sm
fig = sm.qqplot(model.resid,line = 's')
plt.show()
```


    
![png](output_25_0.png)
    


Lastly, we have to check the homoscedasticity assumption. To check the homoscedasticity assumption, you can create a scatterplot of the fitted values and residuals. If the plot resembles a random cloud (i.e., the residuals are scattered randomly), then the assumption is likely met.

we can create one scatterplot by using the scatterplot() function from the seaborn package. The first argument is the variable that goes on the x-axis. The second argument is the variable that goes on the y-axis


```python
# Import matplotlib
import matplotlib.pyplot as plt
fig = sns.scatterplot(x=fitted_values, y=residuals)

# Add reference line at residuals = 0
fig.axhline(0)

# Set x-axis and y-axis labels
fig.set_xlabel("Fitted Values")
fig.set_ylabel("Residuals")

# Show the plot
plt.show()
```


    
![png](output_27_0.png)
    



```python

```
