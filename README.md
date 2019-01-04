
# Scenario: Broken Student Code

A student comes to you with the following notebook while learning about regression. They can't get the code to work. Fix the code, and use this as an opportunity to teach them about the causes of (and solutions to) any underlying mistakes. 

At minimum, you should get the regression to run, and then help them interpret the results. The interviewer will also walk through some follow-up questions, which may require further code to answer. 



```python
import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import statsmodels.api as sm
```


```python
df = pd.read_csv('interview_dataset.csv')
```


```python
target = df.target
predictors = df.drop('target', axis=1, inplace=False)
```


```python
regression_model = sm.OLS(predictors, df)
```
