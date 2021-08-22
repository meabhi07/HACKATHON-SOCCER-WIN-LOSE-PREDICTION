IMPORTING NECESSARY DEPENDENCIES


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
```

IMPORTING TRAINING DATASET


```python
df = pd.read_csv('train.csv')
df = df.drop(['date', 'league', 'Team 1', 'Team2','importance1', 'importance2', 'score1', 'score2', 'xg1', 'xg2', 'nsxg1',
       'nsxg2', 'adj_score1', 'adj_score2'], axis=1)
df.head(2)
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
      <th>season</th>
      <th>league_id</th>
      <th>SPI1</th>
      <th>SPI2</th>
      <th>proj_score1</th>
      <th>proj_score2</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019</td>
      <td>1979</td>
      <td>48.22</td>
      <td>37.83</td>
      <td>1.75</td>
      <td>0.84</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019</td>
      <td>1979</td>
      <td>39.81</td>
      <td>60.08</td>
      <td>1.22</td>
      <td>1.89</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



IMPORTING TEST DATASET


```python
df_test = pd.read_csv('test.csv')

df_test = df_test.drop(['date', 'league', 'Team 1', 'Team2','importance1', 'importance2', 'score1', 'score2', 'xg1', 'xg2', 'nsxg1',
       'nsxg2', 'adj_score1', 'adj_score2'], axis=1)
df_test.head(2)
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
      <th>season</th>
      <th>league_id</th>
      <th>SPI1</th>
      <th>SPI2</th>
      <th>proj_score1</th>
      <th>proj_score2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021</td>
      <td>2411</td>
      <td>79.65</td>
      <td>74.06</td>
      <td>1.67</td>
      <td>1.19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021</td>
      <td>2411</td>
      <td>74.19</td>
      <td>71.14</td>
      <td>1.35</td>
      <td>0.98</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['season', 'league_id', 'SPI1', 'SPI2', 'proj_score1', 'proj_score2',
           'Outcome'],
          dtype='object')



DEFINING NUMPY ARRAY FOR TRAINING DATA


```python
X = df.iloc[:,0:6].values
y = df['Outcome'].values
X
```




    array([[2.019e+03, 1.979e+03, 4.822e+01, 3.783e+01, 1.750e+00, 8.400e-01],
           [2.019e+03, 1.979e+03, 3.981e+01, 6.008e+01, 1.220e+00, 1.890e+00],
           [2.019e+03, 1.979e+03, 6.559e+01, 3.999e+01, 2.580e+00, 6.200e-01],
           ...,
           [2.021e+03, 1.983e+03, 1.297e+01, 2.359e+01, 1.050e+00, 1.500e+00],
           [2.021e+03, 1.983e+03, 1.176e+01, 1.807e+01, 1.220e+00, 1.460e+00],
           [2.021e+03, 1.983e+03, 2.589e+01, 1.075e+01, 1.410e+00, 5.900e-01]])



SCALING THE TRAINING DATA


```python
scaler = StandardScaler().fit(X)
X  = scaler.transform(X)
X
```




    array([[-4.18960998, -0.27624413,  0.42319587, -0.14471537,  0.77524312,
            -0.99580336],
           [-4.18960998, -0.27624413, -0.03470557,  1.06744809, -0.66203631,
             1.87657062],
           [-4.18960998, -0.27624413,  1.36894476, -0.02704017,  3.02607696,
            -1.5976341 ],
           ...,
           [ 0.34778792, -0.27237294, -1.49607   , -0.92049998, -1.12305047,
             0.80968886],
           [ 0.34778792, -0.27237294, -1.56195118, -1.22122547, -0.66203631,
             0.70026509],
           [ 0.34778792, -0.27237294, -0.79261142, -1.62001363, -0.14678519,
            -1.67970192]])



DEFINING NUMPY ARRAY FOR TEST DATA AND SCALING THE TEST DATA


```python
X_test = df_test.iloc[:,0:6].values
scaler = StandardScaler().fit(X_test)
X_test  = scaler.transform(X_test)
X_test
```




    array([[ 0.        ,  0.31400768,  1.80397106,  1.53537302,  0.60017971,
            -0.09896393],
           [ 0.        ,  0.31400768,  1.53726483,  1.39265602, -0.28725125,
            -0.68592724],
           [ 0.        ,  0.31400768,  0.8768494 ,  1.40780748, -0.75869895,
             0.73955507],
           ...,
           [ 0.        , -0.31993434, -0.78640098, -0.68407177, -0.39818012,
            -0.2387171 ],
           [ 0.        , -0.31993434, -0.43860824, -0.60244938, -0.34271569,
            -0.9933842 ],
           [ 0.        , -0.31993434, -0.63595132, -0.60587068, -0.2040546 ,
            -0.09896393]])



TRAIN VALIDATION SPLIT


```python
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, test_size=0.3)
```

DEFINING AND INITIAL IMPRESSIONS OF THE MODEL PERFORMANCE


```python
logreg = LogisticRegression()
logreg.fit(X, y)
logreg.score(X_val, y_val)
```




    0.9932825794894761



HYPERPARAMETER TUNING


```python
c_space = np.logspace(-5,8,15)
s = ['newton-cg']
m = [100,200,300]
param_grid = {'C':c_space,
             'max_iter': m,
             'solver': s}
```

GRID SEARCH CV FOR LOGISTIC MODEL WORKING ON UNSEEN TEST DATA


```python
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
logreg_cv.fit(X, y)
logreg_cv.score(X_val, y_val)
pred = logreg_cv.predict(X_test)
```

SAVING THE PREDICTED OUTCOME TO A CSV FILE


```python
submission = pd.DataFrame(pred, columns=['Outcome'])
submission.to_csv('my_submission_file.csv', index=False)
```


```python

```
