
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
df = pd.read_csv('dataset.csv')
```


```python
target = df.target
df.drop('target', axis=1, inplace=True)
```


```python
regression_model = sm.OLS(target, df)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-7-f24169415276> in <module>
    ----> 1 regression_model = sm.OLS(target, df)
    

    ~\AppData\Local\Continuum\anaconda3\lib\site-packages\statsmodels\regression\linear_model.py in __init__(self, endog, exog, missing, hasconst, **kwargs)
        815                  **kwargs):
        816         super(OLS, self).__init__(endog, exog, missing=missing,
    --> 817                                   hasconst=hasconst, **kwargs)
        818         if "weights" in self._init_keys:
        819             self._init_keys.remove("weights")
    

    ~\AppData\Local\Continuum\anaconda3\lib\site-packages\statsmodels\regression\linear_model.py in __init__(self, endog, exog, weights, missing, hasconst, **kwargs)
        661             weights = weights.squeeze()
        662         super(WLS, self).__init__(endog, exog, missing=missing,
    --> 663                                   weights=weights, hasconst=hasconst, **kwargs)
        664         nobs = self.exog.shape[0]
        665         weights = self.weights
    

    ~\AppData\Local\Continuum\anaconda3\lib\site-packages\statsmodels\regression\linear_model.py in __init__(self, endog, exog, **kwargs)
        177     """
        178     def __init__(self, endog, exog, **kwargs):
    --> 179         super(RegressionModel, self).__init__(endog, exog, **kwargs)
        180         self._data_attr.extend(['pinv_wexog', 'wendog', 'wexog', 'weights'])
        181 
    

    ~\AppData\Local\Continuum\anaconda3\lib\site-packages\statsmodels\base\model.py in __init__(self, endog, exog, **kwargs)
        210 
        211     def __init__(self, endog, exog=None, **kwargs):
    --> 212         super(LikelihoodModel, self).__init__(endog, exog, **kwargs)
        213         self.initialize()
        214 
    

    ~\AppData\Local\Continuum\anaconda3\lib\site-packages\statsmodels\base\model.py in __init__(self, endog, exog, **kwargs)
         62         hasconst = kwargs.pop('hasconst', None)
         63         self.data = self._handle_data(endog, exog, missing, hasconst,
    ---> 64                                       **kwargs)
         65         self.k_constant = self.data.k_constant
         66         self.exog = self.data.exog
    

    ~\AppData\Local\Continuum\anaconda3\lib\site-packages\statsmodels\base\model.py in _handle_data(self, endog, exog, missing, hasconst, **kwargs)
         85 
         86     def _handle_data(self, endog, exog, missing, hasconst, **kwargs):
    ---> 87         data = handle_data(endog, exog, missing, hasconst, **kwargs)
         88         # kwargs arrays could have changed, easier to just attach here
         89         for key in kwargs:
    

    ~\AppData\Local\Continuum\anaconda3\lib\site-packages\statsmodels\base\data.py in handle_data(endog, exog, missing, hasconst, **kwargs)
        631     klass = handle_data_class_factory(endog, exog)
        632     return klass(endog, exog=exog, missing=missing, hasconst=hasconst,
    --> 633                  **kwargs)
    

    ~\AppData\Local\Continuum\anaconda3\lib\site-packages\statsmodels\base\data.py in __init__(self, endog, exog, missing, hasconst, **kwargs)
         74             self.orig_endog = endog
         75             self.orig_exog = exog
    ---> 76             self.endog, self.exog = self._convert_endog_exog(endog, exog)
         77 
         78         # this has side-effects, attaches k_constant and const_idx
    

    ~\AppData\Local\Continuum\anaconda3\lib\site-packages\statsmodels\base\data.py in _convert_endog_exog(self, endog, exog)
        472         exog = exog if exog is None else np.asarray(exog)
        473         if endog.dtype == object or exog is not None and exog.dtype == object:
    --> 474             raise ValueError("Pandas data cast to numpy dtype of object. "
        475                              "Check input data with np.asarray(data).")
        476         return super(PandasData, self)._convert_endog_exog(endog, exog)
    

    ValueError: Pandas data cast to numpy dtype of object. Check input data with np.asarray(data).

