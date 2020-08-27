---
title: "MSDS - ML Indiv"
date: 2019-08-15
tags: [machine learning, adaboost, bankruptcy]
header:
  image: "/images/brankruptcy_adaboost/brankruptcy_adaboost.png"
excerpt: "Data Wrangling, Data Science, Messy Data"
mathjax: "true"
---


# Banker Hamlet Asks To B or Not to B. Predicting Bankruptcy using the Financial Ratios

**Prepared by:** Radney Racela

## Highlights

Bankruptcy is the legal proceeding involving a person or business that is unable to repay outstanding debts. Some of the companies that filed from bankruptcy are Rockport in May 2018, Nine West in April 2018 and Payless in February 2019. In the Philippines, Hanjin has filed for bankruptcy that affected the majority of the banks amounting to 400M USD. Being able to predict that bankruptcy of a company is a crucial task to reduce the risk among banks. The data on Polish companies bankruptcy data from the UCI ML Repository will be used to train the model and detect the predictors of a bankruptcy. Various techniques was used althroughout the report. 

Since we are using imbalanced dataset given that there is a large difference on companies that go bankrupt or not, we used a resampling technique called SMOTE(Synthetic Minority Over-sampling Technique). SMOTE is a resampling technique where it synthesizes the minority class to make the data balanced. This reduces that bias on the majority class and can also magnify the relationship between features. Afterwards, we tried 4 ensemble models and boosting algorithm namely, Decision Tree, Random Forest, Gradient Boosting Method and Adaptive Boost Method. Boosting methods are algorithm that converts weak learners to strong one learning from the iteration of the previous one. 

As shown in the results, Adaptive Boost result to a higher accuracy comparedto the ordinary Decision Tree. Comparing the rest, the accuracy is similar across three. Ultimately, we have chosen GBM among the tested ensemble for its accuracy.

## Data Source

The data came from UCI Machine learning repository entitled, **Polish companies bankruptcy data**  
https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data

The dataset is about bankruptcy prediction of Polish companies. The data was collected from Emerging Markets Information Service (EMIS, [Web Link]), which is a database containing information on emerging markets around the world. The bankrupt companies were analyzed in the period 2000-2012, while the still operating companies were evaluated from 2007 to 2013. The following financial ratios was used in the report.

|Attribute | Description |
| - | - |
| X1	| net profit / total assets |
| X2	|	total liabilities / total assets |
| X3	|	working capital / total assets |
| X4	|	current assets / short-term liabilities |
| X5	|[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365 |
| X6	|retained earnings / total assets |
| X7	|EBIT / total assets |
| X8	|book value of equity / total liabilities |
| X9	|sales / total assets |
| X10	|equity / total assets |
| X11	|(gross profit + extraordinary items + financial expenses) / total assets |
| X12	|gross profit / short-term liabilities |
| X13	|(gross profit + depreciation) / sales |
| X14	|(gross profit + interest) / total assets |
| X15	|(total liabilities * 365) / (gross profit + depreciation) |
| X16	|(gross profit + depreciation) / total liabilities |
| X17	|total assets / total liabilities |
| X18	|gross profit / total assets |
| X19	|gross profit / sales |
| X20	|(inventory * 365) / sales |
| X21	|sales (n) / sales (n-1) |
| X22	|profit on operating activities / total assets |
| X23	|net profit / sales |
| X24	|gross profit (in 3 years) / total assets |
| X25	|(equity - share capital) / total assets |
| X26	|(net profit + depreciation) / total liabilities |
| X27	|profit on operating activities / financial expenses |
| X28	|working capital / fixed assets |
| X29	|logarithm of total assets |
| X30	|(total liabilities - cash) / sales |
| X31	|(gross profit + interest) / sales |
| X32	|(current liabilities * 365) / cost of products sold |
| X33	|operating expenses / short-term liabilities |
| X34	|operating expenses / total liabilities |
| X35	|profit on sales / total assets |
| X36	|total sales / total assets |
| X37	|(current assets - inventories) / long-term liabilities |
| X38	|constant capital / total assets |
| X39	|profit on sales / sales |
| X40	|(current assets - inventory - receivables) / short-term liabilities |
| X41	|total liabilities / ((profit on operating activities + depreciation) * (12/365)) |
| X42	|profit on operating activities / sales |
| X43	|rotation receivables + inventory turnover in days |
| X44	|(receivables * 365) / sales |
| X45	|net profit / inventory |
| X46	|(current assets - inventory) / short-term liabilities |
| X47	|(inventory * 365) / cost of products sold |
| X48	|EBITDA (profit on operating activities - depreciation) / total assets |
| X49	|EBITDA (profit on operating activities - depreciation) / sales |
| X50	|current assets / total liabilities |
| X51	|short-term liabilities / total assets |
| X52	|(short-term liabilities * 365) / cost of products sold) |
| X53	|equity / fixed assets |
| X54	|constant capital / fixed assets |
| X55	|working capital |
| X56	|(sales - cost of products sold) / sales |
| X57	|(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation) |
| X58	|total costs /total sales |
| X59	|long-term liabilities / equity |
| X60	|sales / inventory |
| X61	|sales / receivables |
| X62	|(short-term liabilities *365) / sales |
| X63	|sales / short-term liabilities |
| X64	|sales / fixed assets



## Methods and Discussion

We load the data to a dataframe using the arff library. We then identified correlations, removed the nulls before the modeling. The dataset was then modeled using 4 methods and compared to each other. The final model is then validated with confusion matrix.

### Library

Load the followign libraries:
* `arff` from `scipy.io` (open arff files)
* `numpy` - for numerical maninupation
* `pandas` - for storing data and manipulation
* `matplotlib.pyplot` and `seaborn` - for visualization
* `judas` - for specialized modeling 


```python
from scipy.io import arff
# Data Manipulation
from collections import Counter
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# Load Judas Libraries
from judas.regression.automate import Judas as JudasRegressor
from judas.classification.automate import Judas as JudasClassifier
from judas.automate import General

# Import Dependencies
%matplotlib inline

gen = General()
```

    C:\Users\Rad\Anaconda3\lib\site-packages\tqdm\autonotebook\__init__.py:14: TqdmExperimentalWarning: Using `tqdm.autonotebook.tqdm` in notebook mode. Use `tqdm.tqdm` instead to force console mode (e.g. in jupyter console)
      " (e.g. in jupyter console)", TqdmExperimentalWarning)
    

### Load Dataset

Load the files from UCI.


```python
df=[]
for w in range(1,6):
    data = arff.loadarff(str(w)+'year.arff')
    df.append(pd.DataFrame(data[0]))
    df[w-1].info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7027 entries, 0 to 7026
    Data columns (total 65 columns):
    Attr1     7024 non-null float64
    Attr2     7024 non-null float64
    Attr3     7024 non-null float64
    Attr4     6997 non-null float64
    Attr5     7019 non-null float64
    Attr6     7024 non-null float64
    Attr7     7024 non-null float64
    Attr8     7002 non-null float64
    Attr9     7026 non-null float64
    Attr10    7024 non-null float64
    Attr11    6988 non-null float64
    Attr12    6997 non-null float64
    Attr13    7027 non-null float64
    Attr14    7024 non-null float64
    Attr15    7025 non-null float64
    Attr16    7002 non-null float64
    Attr17    7002 non-null float64
    Attr18    7024 non-null float64
    Attr19    7027 non-null float64
    Attr20    7027 non-null float64
    Attr21    5405 non-null float64
    Attr22    7024 non-null float64
    Attr23    7027 non-null float64
    Attr24    6903 non-null float64
    Attr25    7024 non-null float64
    Attr26    7002 non-null float64
    Attr27    6716 non-null float64
    Attr28    6993 non-null float64
    Attr29    7024 non-null float64
    Attr30    7027 non-null float64
    Attr31    7027 non-null float64
    Attr32    6989 non-null float64
    Attr33    6997 non-null float64
    Attr34    7002 non-null float64
    Attr35    7024 non-null float64
    Attr36    7024 non-null float64
    Attr37    4287 non-null float64
    Attr38    7024 non-null float64
    Attr39    7027 non-null float64
    Attr40    6997 non-null float64
    Attr41    6943 non-null float64
    Attr42    7027 non-null float64
    Attr43    7027 non-null float64
    Attr44    7027 non-null float64
    Attr45    6893 non-null float64
    Attr46    6996 non-null float64
    Attr47    6998 non-null float64
    Attr48    7024 non-null float64
    Attr49    7027 non-null float64
    Attr50    7002 non-null float64
    Attr51    7024 non-null float64
    Attr52    6998 non-null float64
    Attr53    6993 non-null float64
    Attr54    6993 non-null float64
    Attr55    7027 non-null float64
    Attr56    7027 non-null float64
    Attr57    7026 non-null float64
    Attr58    7027 non-null float64
    Attr59    7026 non-null float64
    Attr60    6892 non-null float64
    Attr61    7005 non-null float64
    Attr62    7027 non-null float64
    Attr63    6997 non-null float64
    Attr64    6993 non-null float64
    class     7027 non-null object
    dtypes: float64(64), object(1)
    memory usage: 3.5+ MB
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10173 entries, 0 to 10172
    Data columns (total 65 columns):
    Attr1     10172 non-null float64
    Attr2     10172 non-null float64
    Attr3     10172 non-null float64
    Attr4     10151 non-null float64
    Attr5     10149 non-null float64
    Attr6     10172 non-null float64
    Attr7     10172 non-null float64
    Attr8     10155 non-null float64
    Attr9     10169 non-null float64
    Attr10    10172 non-null float64
    Attr11    10172 non-null float64
    Attr12    10151 non-null float64
    Attr13    10110 non-null float64
    Attr14    10172 non-null float64
    Attr15    10161 non-null float64
    Attr16    10154 non-null float64
    Attr17    10155 non-null float64
    Attr18    10172 non-null float64
    Attr19    10109 non-null float64
    Attr20    10110 non-null float64
    Attr21    7009 non-null float64
    Attr22    10172 non-null float64
    Attr23    10110 non-null float64
    Attr24    9948 non-null float64
    Attr25    10172 non-null float64
    Attr26    10154 non-null float64
    Attr27    9467 non-null float64
    Attr28    9961 non-null float64
    Attr29    10172 non-null float64
    Attr30    10110 non-null float64
    Attr31    10110 non-null float64
    Attr32    10086 non-null float64
    Attr33    10151 non-null float64
    Attr34    10155 non-null float64
    Attr35    10172 non-null float64
    Attr36    10172 non-null float64
    Attr37    5655 non-null float64
    Attr38    10172 non-null float64
    Attr39    10110 non-null float64
    Attr40    10151 non-null float64
    Attr41    9976 non-null float64
    Attr42    10110 non-null float64
    Attr43    10110 non-null float64
    Attr44    10110 non-null float64
    Attr45    9632 non-null float64
    Attr46    10151 non-null float64
    Attr47    10099 non-null float64
    Attr48    10171 non-null float64
    Attr49    10110 non-null float64
    Attr50    10155 non-null float64
    Attr51    10172 non-null float64
    Attr52    10099 non-null float64
    Attr53    9961 non-null float64
    Attr54    9961 non-null float64
    Attr55    10172 non-null float64
    Attr56    10110 non-null float64
    Attr57    10171 non-null float64
    Attr58    10134 non-null float64
    Attr59    10171 non-null float64
    Attr60    9630 non-null float64
    Attr61    10157 non-null float64
    Attr62    10110 non-null float64
    Attr63    10151 non-null float64
    Attr64    9961 non-null float64
    class     10173 non-null object
    dtypes: float64(64), object(1)
    memory usage: 5.0+ MB
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10503 entries, 0 to 10502
    Data columns (total 65 columns):
    Attr1     10503 non-null float64
    Attr2     10503 non-null float64
    Attr3     10503 non-null float64
    Attr4     10485 non-null float64
    Attr5     10478 non-null float64
    Attr6     10503 non-null float64
    Attr7     10503 non-null float64
    Attr8     10489 non-null float64
    Attr9     10500 non-null float64
    Attr10    10503 non-null float64
    Attr11    10503 non-null float64
    Attr12    10485 non-null float64
    Attr13    10460 non-null float64
    Attr14    10503 non-null float64
    Attr15    10495 non-null float64
    Attr16    10489 non-null float64
    Attr17    10489 non-null float64
    Attr18    10503 non-null float64
    Attr19    10460 non-null float64
    Attr20    10460 non-null float64
    Attr21    9696 non-null float64
    Attr22    10503 non-null float64
    Attr23    10460 non-null float64
    Attr24    10276 non-null float64
    Attr25    10503 non-null float64
    Attr26    10489 non-null float64
    Attr27    9788 non-null float64
    Attr28    10275 non-null float64
    Attr29    10503 non-null float64
    Attr30    10460 non-null float64
    Attr31    10460 non-null float64
    Attr32    10402 non-null float64
    Attr33    10485 non-null float64
    Attr34    10489 non-null float64
    Attr35    10503 non-null float64
    Attr36    10503 non-null float64
    Attr37    5767 non-null float64
    Attr38    10503 non-null float64
    Attr39    10460 non-null float64
    Attr40    10485 non-null float64
    Attr41    10301 non-null float64
    Attr42    10460 non-null float64
    Attr43    10460 non-null float64
    Attr44    10460 non-null float64
    Attr45    9912 non-null float64
    Attr46    10485 non-null float64
    Attr47    10417 non-null float64
    Attr48    10503 non-null float64
    Attr49    10460 non-null float64
    Attr50    10489 non-null float64
    Attr51    10503 non-null float64
    Attr52    10417 non-null float64
    Attr53    10275 non-null float64
    Attr54    10275 non-null float64
    Attr55    10503 non-null float64
    Attr56    10460 non-null float64
    Attr57    10503 non-null float64
    Attr58    10474 non-null float64
    Attr59    10503 non-null float64
    Attr60    9911 non-null float64
    Attr61    10486 non-null float64
    Attr62    10460 non-null float64
    Attr63    10485 non-null float64
    Attr64    10275 non-null float64
    class     10503 non-null object
    dtypes: float64(64), object(1)
    memory usage: 5.2+ MB
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9792 entries, 0 to 9791
    Data columns (total 65 columns):
    Attr1     9791 non-null float64
    Attr2     9791 non-null float64
    Attr3     9791 non-null float64
    Attr4     9749 non-null float64
    Attr5     9771 non-null float64
    Attr6     9791 non-null float64
    Attr7     9791 non-null float64
    Attr8     9773 non-null float64
    Attr9     9792 non-null float64
    Attr10    9791 non-null float64
    Attr11    9791 non-null float64
    Attr12    9749 non-null float64
    Attr13    9771 non-null float64
    Attr14    9791 non-null float64
    Attr15    9784 non-null float64
    Attr16    9773 non-null float64
    Attr17    9773 non-null float64
    Attr18    9791 non-null float64
    Attr19    9771 non-null float64
    Attr20    9771 non-null float64
    Attr21    9634 non-null float64
    Attr22    9791 non-null float64
    Attr23    9771 non-null float64
    Attr24    9581 non-null float64
    Attr25    9791 non-null float64
    Attr26    9773 non-null float64
    Attr27    9151 non-null float64
    Attr28    9561 non-null float64
    Attr29    9791 non-null float64
    Attr30    9771 non-null float64
    Attr31    9771 non-null float64
    Attr32    9696 non-null float64
    Attr33    9749 non-null float64
    Attr34    9773 non-null float64
    Attr35    9791 non-null float64
    Attr36    9791 non-null float64
    Attr37    5350 non-null float64
    Attr38    9791 non-null float64
    Attr39    9771 non-null float64
    Attr40    9749 non-null float64
    Attr41    9605 non-null float64
    Attr42    9771 non-null float64
    Attr43    9771 non-null float64
    Attr44    9771 non-null float64
    Attr45    9179 non-null float64
    Attr46    9749 non-null float64
    Attr47    9719 non-null float64
    Attr48    9791 non-null float64
    Attr49    9771 non-null float64
    Attr50    9773 non-null float64
    Attr51    9791 non-null float64
    Attr52    9716 non-null float64
    Attr53    9561 non-null float64
    Attr54    9561 non-null float64
    Attr55    9792 non-null float64
    Attr56    9771 non-null float64
    Attr57    9791 non-null float64
    Attr58    9776 non-null float64
    Attr59    9791 non-null float64
    Attr60    9178 non-null float64
    Attr61    9760 non-null float64
    Attr62    9771 non-null float64
    Attr63    9749 non-null float64
    Attr64    9561 non-null float64
    class     9792 non-null object
    dtypes: float64(64), object(1)
    memory usage: 4.9+ MB
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5910 entries, 0 to 5909
    Data columns (total 65 columns):
    Attr1     5907 non-null float64
    Attr2     5907 non-null float64
    Attr3     5907 non-null float64
    Attr4     5889 non-null float64
    Attr5     5899 non-null float64
    Attr6     5907 non-null float64
    Attr7     5907 non-null float64
    Attr8     5892 non-null float64
    Attr9     5909 non-null float64
    Attr10    5907 non-null float64
    Attr11    5907 non-null float64
    Attr12    5889 non-null float64
    Attr13    5910 non-null float64
    Attr14    5907 non-null float64
    Attr15    5904 non-null float64
    Attr16    5892 non-null float64
    Attr17    5892 non-null float64
    Attr18    5907 non-null float64
    Attr19    5910 non-null float64
    Attr20    5910 non-null float64
    Attr21    5807 non-null float64
    Attr22    5907 non-null float64
    Attr23    5910 non-null float64
    Attr24    5775 non-null float64
    Attr25    5907 non-null float64
    Attr26    5892 non-null float64
    Attr27    5519 non-null float64
    Attr28    5803 non-null float64
    Attr29    5907 non-null float64
    Attr30    5910 non-null float64
    Attr31    5910 non-null float64
    Attr32    5864 non-null float64
    Attr33    5889 non-null float64
    Attr34    5892 non-null float64
    Attr35    5907 non-null float64
    Attr36    5907 non-null float64
    Attr37    3362 non-null float64
    Attr38    5907 non-null float64
    Attr39    5910 non-null float64
    Attr40    5889 non-null float64
    Attr41    5826 non-null float64
    Attr42    5910 non-null float64
    Attr43    5910 non-null float64
    Attr44    5910 non-null float64
    Attr45    5642 non-null float64
    Attr46    5889 non-null float64
    Attr47    5875 non-null float64
    Attr48    5907 non-null float64
    Attr49    5910 non-null float64
    Attr50    5892 non-null float64
    Attr51    5907 non-null float64
    Attr52    5874 non-null float64
    Attr53    5803 non-null float64
    Attr54    5803 non-null float64
    Attr55    5910 non-null float64
    Attr56    5910 non-null float64
    Attr57    5907 non-null float64
    Attr58    5910 non-null float64
    Attr59    5907 non-null float64
    Attr60    5642 non-null float64
    Attr61    5895 non-null float64
    Attr62    5910 non-null float64
    Attr63    5889 non-null float64
    Attr64    5803 non-null float64
    class     5910 non-null object
    dtypes: float64(64), object(1)
    memory usage: 2.9+ MB
    

Convert class column to int from binary.


```python
for w in range(0,5):
    df[w]['class'] = df[w]['class'].astype(int)
```

Merge All Dataset into one big DataFrame


```python
df1 = df[0]
for w in range(1,5):
    df1 = pd.concat([df1,df[w]])
df1.head()
    
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
      <th>Attr1</th>
      <th>Attr2</th>
      <th>Attr3</th>
      <th>Attr4</th>
      <th>Attr5</th>
      <th>Attr6</th>
      <th>Attr7</th>
      <th>Attr8</th>
      <th>Attr9</th>
      <th>Attr10</th>
      <th>...</th>
      <th>Attr56</th>
      <th>Attr57</th>
      <th>Attr58</th>
      <th>Attr59</th>
      <th>Attr60</th>
      <th>Attr61</th>
      <th>Attr62</th>
      <th>Attr63</th>
      <th>Attr64</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.200550</td>
      <td>0.37951</td>
      <td>0.39641</td>
      <td>2.0472</td>
      <td>32.3510</td>
      <td>0.38825</td>
      <td>0.249760</td>
      <td>1.33050</td>
      <td>1.1389</td>
      <td>0.50494</td>
      <td>...</td>
      <td>0.121960</td>
      <td>0.39718</td>
      <td>0.87804</td>
      <td>0.001924</td>
      <td>8.4160</td>
      <td>5.1372</td>
      <td>82.658</td>
      <td>4.4158</td>
      <td>7.4277</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.209120</td>
      <td>0.49988</td>
      <td>0.47225</td>
      <td>1.9447</td>
      <td>14.7860</td>
      <td>0.00000</td>
      <td>0.258340</td>
      <td>0.99601</td>
      <td>1.6996</td>
      <td>0.49788</td>
      <td>...</td>
      <td>0.121300</td>
      <td>0.42002</td>
      <td>0.85300</td>
      <td>0.000000</td>
      <td>4.1486</td>
      <td>3.2732</td>
      <td>107.350</td>
      <td>3.4000</td>
      <td>60.9870</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.248660</td>
      <td>0.69592</td>
      <td>0.26713</td>
      <td>1.5548</td>
      <td>-1.1523</td>
      <td>0.00000</td>
      <td>0.309060</td>
      <td>0.43695</td>
      <td>1.3090</td>
      <td>0.30408</td>
      <td>...</td>
      <td>0.241140</td>
      <td>0.81774</td>
      <td>0.76599</td>
      <td>0.694840</td>
      <td>4.9909</td>
      <td>3.9510</td>
      <td>134.270</td>
      <td>2.7185</td>
      <td>5.2078</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.081483</td>
      <td>0.30734</td>
      <td>0.45879</td>
      <td>2.4928</td>
      <td>51.9520</td>
      <td>0.14988</td>
      <td>0.092704</td>
      <td>1.86610</td>
      <td>1.0571</td>
      <td>0.57353</td>
      <td>...</td>
      <td>0.054015</td>
      <td>0.14207</td>
      <td>0.94598</td>
      <td>0.000000</td>
      <td>4.5746</td>
      <td>3.6147</td>
      <td>86.435</td>
      <td>4.2228</td>
      <td>5.5497</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.187320</td>
      <td>0.61323</td>
      <td>0.22960</td>
      <td>1.4063</td>
      <td>-7.3128</td>
      <td>0.18732</td>
      <td>0.187320</td>
      <td>0.63070</td>
      <td>1.1559</td>
      <td>0.38677</td>
      <td>...</td>
      <td>0.134850</td>
      <td>0.48431</td>
      <td>0.86515</td>
      <td>0.124440</td>
      <td>6.3985</td>
      <td>4.3158</td>
      <td>127.210</td>
      <td>2.8692</td>
      <td>7.8980</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 65 columns</p>
</div>



Check the PCC for the target variable.


```python
gen.pcc(df1['class'])
```

    Counter({0: 41314, 1: 2091})
    State Count:        0
    0  41314
    1   2091
    
    1.25 * Proportion Chance Criterion: 113.53664437709668%
    


![png](/images/brankruptcy_adaboost/brankruptcy_adaboost_17_1.png)


Check Distribution of each attribute / financial ratio


```python
df1.hist(figsize=(15,15));
```


![png](/images/brankruptcy_adaboost/brankruptcy_adaboost_19_0.png)


Check correlation between attributes.


```python
fig, ax = plt.subplots(figsize=(15,15)) 
sns.heatmap(df1.corr(), cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = False, vmax = 0.6,ax=ax)
ax.set_title('Correlation Heatmap');
```


![png](/images/brankruptcy_adaboost/brankruptcy_adaboost_21_0.png)


Set threshold for the correlation. Any correlation greater than 0.9 will be candidate for deletion.


```python
threshold = 0.9
corr_matrix = df1.corr().abs()
corr_matrix.head()
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
      <th>Attr1</th>
      <th>Attr2</th>
      <th>Attr3</th>
      <th>Attr4</th>
      <th>Attr5</th>
      <th>Attr6</th>
      <th>Attr7</th>
      <th>Attr8</th>
      <th>Attr9</th>
      <th>Attr10</th>
      <th>...</th>
      <th>Attr56</th>
      <th>Attr57</th>
      <th>Attr58</th>
      <th>Attr59</th>
      <th>Attr60</th>
      <th>Attr61</th>
      <th>Attr62</th>
      <th>Attr63</th>
      <th>Attr64</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Attr1</th>
      <td>1.000000</td>
      <td>0.104146</td>
      <td>0.058037</td>
      <td>0.000855</td>
      <td>0.002742</td>
      <td>0.315139</td>
      <td>0.408428</td>
      <td>0.000184</td>
      <td>0.225720</td>
      <td>0.240254</td>
      <td>...</td>
      <td>0.000044</td>
      <td>0.007764</td>
      <td>0.000302</td>
      <td>0.000220</td>
      <td>0.000213</td>
      <td>0.000582</td>
      <td>0.000325</td>
      <td>0.008387</td>
      <td>0.069029</td>
      <td>0.026649</td>
    </tr>
    <tr>
      <th>Attr2</th>
      <td>0.104146</td>
      <td>1.000000</td>
      <td>0.926983</td>
      <td>0.001589</td>
      <td>0.036344</td>
      <td>0.842916</td>
      <td>0.102020</td>
      <td>0.002385</td>
      <td>0.016594</td>
      <td>0.409341</td>
      <td>...</td>
      <td>0.000235</td>
      <td>0.000793</td>
      <td>0.000270</td>
      <td>0.000617</td>
      <td>0.000078</td>
      <td>0.001191</td>
      <td>0.046871</td>
      <td>0.003339</td>
      <td>0.015649</td>
      <td>0.035236</td>
    </tr>
    <tr>
      <th>Attr3</th>
      <td>0.058037</td>
      <td>0.926983</td>
      <td>1.000000</td>
      <td>0.002335</td>
      <td>0.038900</td>
      <td>0.760215</td>
      <td>0.033521</td>
      <td>0.001267</td>
      <td>0.002981</td>
      <td>0.369558</td>
      <td>...</td>
      <td>0.000017</td>
      <td>0.000779</td>
      <td>0.000034</td>
      <td>0.000247</td>
      <td>0.000377</td>
      <td>0.000139</td>
      <td>0.050162</td>
      <td>0.004044</td>
      <td>0.000909</td>
      <td>0.035128</td>
    </tr>
    <tr>
      <th>Attr4</th>
      <td>0.000855</td>
      <td>0.001589</td>
      <td>0.002335</td>
      <td>1.000000</td>
      <td>0.001852</td>
      <td>0.000050</td>
      <td>0.000167</td>
      <td>0.598635</td>
      <td>0.000392</td>
      <td>0.001668</td>
      <td>...</td>
      <td>0.000281</td>
      <td>0.000125</td>
      <td>0.000289</td>
      <td>0.000266</td>
      <td>0.001671</td>
      <td>0.004387</td>
      <td>0.000704</td>
      <td>0.039447</td>
      <td>0.000122</td>
      <td>0.001652</td>
    </tr>
    <tr>
      <th>Attr5</th>
      <td>0.002742</td>
      <td>0.036344</td>
      <td>0.038900</td>
      <td>0.001852</td>
      <td>1.000000</td>
      <td>0.029307</td>
      <td>0.001491</td>
      <td>0.001631</td>
      <td>0.000796</td>
      <td>0.014709</td>
      <td>...</td>
      <td>0.000031</td>
      <td>0.000009</td>
      <td>0.000029</td>
      <td>0.000067</td>
      <td>0.000087</td>
      <td>0.000007</td>
      <td>0.002639</td>
      <td>0.000690</td>
      <td>0.000198</td>
      <td>0.001327</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 65 columns</p>
</div>




```python
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head()
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
      <th>Attr1</th>
      <th>Attr2</th>
      <th>Attr3</th>
      <th>Attr4</th>
      <th>Attr5</th>
      <th>Attr6</th>
      <th>Attr7</th>
      <th>Attr8</th>
      <th>Attr9</th>
      <th>Attr10</th>
      <th>...</th>
      <th>Attr56</th>
      <th>Attr57</th>
      <th>Attr58</th>
      <th>Attr59</th>
      <th>Attr60</th>
      <th>Attr61</th>
      <th>Attr62</th>
      <th>Attr63</th>
      <th>Attr64</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Attr1</th>
      <td>NaN</td>
      <td>0.104146</td>
      <td>0.058037</td>
      <td>0.000855</td>
      <td>0.002742</td>
      <td>0.315139</td>
      <td>0.408428</td>
      <td>0.000184</td>
      <td>0.225720</td>
      <td>0.240254</td>
      <td>...</td>
      <td>0.000044</td>
      <td>0.007764</td>
      <td>0.000302</td>
      <td>0.000220</td>
      <td>0.000213</td>
      <td>0.000582</td>
      <td>0.000325</td>
      <td>0.008387</td>
      <td>0.069029</td>
      <td>0.026649</td>
    </tr>
    <tr>
      <th>Attr2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.926983</td>
      <td>0.001589</td>
      <td>0.036344</td>
      <td>0.842916</td>
      <td>0.102020</td>
      <td>0.002385</td>
      <td>0.016594</td>
      <td>0.409341</td>
      <td>...</td>
      <td>0.000235</td>
      <td>0.000793</td>
      <td>0.000270</td>
      <td>0.000617</td>
      <td>0.000078</td>
      <td>0.001191</td>
      <td>0.046871</td>
      <td>0.003339</td>
      <td>0.015649</td>
      <td>0.035236</td>
    </tr>
    <tr>
      <th>Attr3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.002335</td>
      <td>0.038900</td>
      <td>0.760215</td>
      <td>0.033521</td>
      <td>0.001267</td>
      <td>0.002981</td>
      <td>0.369558</td>
      <td>...</td>
      <td>0.000017</td>
      <td>0.000779</td>
      <td>0.000034</td>
      <td>0.000247</td>
      <td>0.000377</td>
      <td>0.000139</td>
      <td>0.050162</td>
      <td>0.004044</td>
      <td>0.000909</td>
      <td>0.035128</td>
    </tr>
    <tr>
      <th>Attr4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.001852</td>
      <td>0.000050</td>
      <td>0.000167</td>
      <td>0.598635</td>
      <td>0.000392</td>
      <td>0.001668</td>
      <td>...</td>
      <td>0.000281</td>
      <td>0.000125</td>
      <td>0.000289</td>
      <td>0.000266</td>
      <td>0.001671</td>
      <td>0.004387</td>
      <td>0.000704</td>
      <td>0.039447</td>
      <td>0.000122</td>
      <td>0.001652</td>
    </tr>
    <tr>
      <th>Attr5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.029307</td>
      <td>0.001491</td>
      <td>0.001631</td>
      <td>0.000796</td>
      <td>0.014709</td>
      <td>...</td>
      <td>0.000031</td>
      <td>0.000009</td>
      <td>0.000029</td>
      <td>0.000067</td>
      <td>0.000087</td>
      <td>0.000007</td>
      <td>0.002639</td>
      <td>0.000690</td>
      <td>0.000198</td>
      <td>0.001327</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 65 columns</p>
</div>




```python
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove.' % (len(to_drop)))

df2 = df1.copy()
df2 = df2.drop(columns = to_drop)
```

    There are 31 columns to remove.
    


```python
df2.shape
```




    (43405, 34)



Check for columns that have majorly null. These columns will be dropped.


```python
df2_missing = (df2.isnull().sum() / len(df2)).sort_values(ascending = False)
df2_missing.head()
```




    Attr37    0.437369
    Attr21    0.134869
    Attr27    0.063679
    Attr60    0.049580
    Attr45    0.049464
    dtype: float64




```python
df3_missing = df2_missing.index[df2_missing > 0.75]

all_missing = list(set(df3_missing))
print('There are %d columns with more than 75%% missing values' % len(all_missing))
```

    There are 0 columns with more than 75% missing values
    

Since majority of the rows in a column is not null, we can drop all rows that has NA instead.


```python
df3 = df2.dropna()
df3.head()
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
      <th>Attr1</th>
      <th>Attr2</th>
      <th>Attr4</th>
      <th>Attr5</th>
      <th>Attr6</th>
      <th>Attr7</th>
      <th>Attr8</th>
      <th>Attr9</th>
      <th>Attr10</th>
      <th>Attr12</th>
      <th>...</th>
      <th>Attr41</th>
      <th>Attr42</th>
      <th>Attr45</th>
      <th>Attr53</th>
      <th>Attr55</th>
      <th>Attr57</th>
      <th>Attr59</th>
      <th>Attr60</th>
      <th>Attr61</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.200550</td>
      <td>0.37951</td>
      <td>2.0472</td>
      <td>32.351</td>
      <td>0.388250</td>
      <td>0.249760</td>
      <td>1.33050</td>
      <td>1.1389</td>
      <td>0.50494</td>
      <td>0.659800</td>
      <td>...</td>
      <td>0.051402</td>
      <td>0.128040</td>
      <td>1.00970</td>
      <td>2.24370</td>
      <td>348690.0000</td>
      <td>0.397180</td>
      <td>0.001924</td>
      <td>8.4160</td>
      <td>5.1372</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.009020</td>
      <td>0.63202</td>
      <td>1.1263</td>
      <td>-37.842</td>
      <td>0.000000</td>
      <td>0.014434</td>
      <td>0.58223</td>
      <td>1.3332</td>
      <td>0.36798</td>
      <td>0.033921</td>
      <td>...</td>
      <td>0.308650</td>
      <td>0.023085</td>
      <td>0.06743</td>
      <td>0.70666</td>
      <td>1.1263</td>
      <td>0.024512</td>
      <td>0.340940</td>
      <td>9.9665</td>
      <td>4.2382</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.266690</td>
      <td>0.34994</td>
      <td>3.0243</td>
      <td>43.087</td>
      <td>0.559830</td>
      <td>0.332070</td>
      <td>1.85770</td>
      <td>1.1268</td>
      <td>0.65006</td>
      <td>1.099300</td>
      <td>...</td>
      <td>0.035883</td>
      <td>0.105480</td>
      <td>0.88253</td>
      <td>7.51920</td>
      <td>5340.0000</td>
      <td>0.410250</td>
      <td>0.073630</td>
      <td>9.5593</td>
      <td>5.6298</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.067731</td>
      <td>0.19885</td>
      <td>2.9576</td>
      <td>90.606</td>
      <td>0.212650</td>
      <td>0.078063</td>
      <td>4.02900</td>
      <td>1.2570</td>
      <td>0.80115</td>
      <td>1.873600</td>
      <td>...</td>
      <td>0.120970</td>
      <td>0.212760</td>
      <td>2.20250</td>
      <td>0.91375</td>
      <td>15132.0000</td>
      <td>0.084542</td>
      <td>0.196190</td>
      <td>8.2122</td>
      <td>2.7917</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-0.029182</td>
      <td>0.21131</td>
      <td>7.5746</td>
      <td>57.844</td>
      <td>0.010387</td>
      <td>-0.034653</td>
      <td>3.73240</td>
      <td>1.0241</td>
      <td>0.78869</td>
      <td>-0.503330</td>
      <td>...</td>
      <td>-0.548790</td>
      <td>-0.064076</td>
      <td>-0.12797</td>
      <td>1.64820</td>
      <td>34549.0000</td>
      <td>-0.037001</td>
      <td>0.180630</td>
      <td>3.4646</td>
      <td>11.3380</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>



After dropping, check the new PCC.


```python
gen.pcc(df3['class'])
```

    Counter({0: 19536, 1: 436})
    State Count:        0
    0  19536
    1    436
    
    1.25 * Proportion Chance Criterion: 119.66150267093337%
    


![png](/images/brankruptcy_adaboost/brankruptcy_adaboost_33_1.png)



```python
target_column = 'class'
df_targetR = df3[target_column]
df_dataR = df3.drop(target_column,axis=1)
```


```python
X = df_dataR
y = df_targetR
```

Since there is an imbalance of data, lets use oversampling via SMOTE. SMOTE synthesises new minority instances between existing (real) minority instances. Imagine that SMOTE draws lines between existing minority instances like this. SMOTE then imagines new, synthetic minority instances somewhere on these lines. [http://rikunert.com/SMOTE_explained]


```python
import imblearn
from imblearn.over_sampling import SMOTE

y_i = y
x_i = X
print('Original dataset shape %s' % Counter(y_i))

X_resampled, y_resampled = SMOTE().fit_resample(x_i, y_i)
print(sorted(Counter(y_resampled).items()))
```

    Original dataset shape Counter({0: 19536, 1: 436})
    [(0, 19536), (1, 19536)]
    


```python
X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
```

After using SMOTE, lets check the new PCC.


```python
gen.pcc(y_resampled)
```

    Counter({0: 19536, 1: 19536})
    State Count:        0
    0  19536
    1  19536
    
    1.25 * Proportion Chance Criterion: 62.5%
    


![png](/images/brankruptcy_adaboost/brankruptcy_adaboost_40_1.png)


Lets check the relationship of each data between after the resampling.


```python
fig, ax = plt.subplots(figsize=(15,15)) 
sns.heatmap(X_resampled.corr(), cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = False, vmax = 0.6,ax=ax)
ax.set_title('Correlation Heatmap');
```


![png](/images/brankruptcy_adaboost/brankruptcy_adaboost_42_0.png)


### Classification Model

Since we will be using ensemble method, we will not do any scaling. Also, we'll use the special function that have internal SMOTE. This way the resampling will only be done on the training set and not on the test set. Models that will be used for comparison are the following:

* `Decision Tree`
* `Random Forest`
* `GBM`
* `AdaBoost with Decision Tree`


```python
from judas.classification.automate import Judas as JudasClassifier

trials = 10
judasc1 = JudasClassifier()
params = [
    {'model': 'ensemble-decisiontreeS', 'trials': trials, 'maxdepth': range(1, 20)},
    {'model': 'ensemble-randomforestS', 'trials': trials, 'n_est': range(1, 20)},
    {'model': 'ensemble-gbmS', 'trials': trials, 'maxdepth': range(1, 10)},
    {'model': 'ensemble-adaboost-treeS', 'trials': trials, 'n_est': range(20, 100, 10)},
]

judasc1.automate(X,y,params)
```

    ensemble-decisiontreeS, max depth=range(1, 20)
    


    HBox(children=(IntProgress(value=0, max=190), HTML(value='')))


    
    ensemble-randomforestS, n estimators=range(1, 20)
    


    HBox(children=(IntProgress(value=0, max=190), HTML(value='')))


    
    ensemble-gbmS, max depth=range(1, 10)
    


    HBox(children=(IntProgress(value=0, max=90), HTML(value='')))


    
    ensemble-adaboost-treeS, max depth=range(20, 100, 10)
    


    HBox(children=(IntProgress(value=0, max=80), HTML(value='')))


    
    


```python
judasc1.score()
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
      <th>Machine Learning Method</th>
      <th>Test Accuracy</th>
      <th>Best Parameter</th>
      <th>Top Predictor Variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Decision Trees</td>
      <td>90.49%</td>
      <td>depth = 19</td>
      <td>Attr25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Random Forest</td>
      <td>95.95%</td>
      <td>n-estimator = 18</td>
      <td>Attr25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gradient Boosting Method</td>
      <td>95.78%</td>
      <td>depth = 9</td>
      <td>Attr25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AdaBoost with Decision Tree Method</td>
      <td>95.24%</td>
      <td>estimator = 90</td>
      <td>Attr32</td>
    </tr>
  </tbody>
</table>
</div>




```python
judasc1.plot_accuracy()
```


![png](ML_Final_IndivProject_Racela_1_files/ML_Final_IndivProject_Racela_1_47_0.png)


We can see that by using boosting method, the accuracy of decision tree increased from 90% to 95%. Based on the results, we can use GBM for our model. Lets validate it.

### Validation

Lets split the dataset from train/test and validation set.


```python
from sklearn.model_selection import train_test_split
X_t, X_Val, y_t, y_Val = train_test_split(X,y, stratify=y,test_size=0.25, random_state=24)
```


```python
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
X_train, X_test, y_train, y_test = train_test_split(X_t,y_t, stratify=y_t,test_size=0.25, random_state=24)

gbrt = GradientBoostingClassifier(max_depth=9, learning_rate=0.1, random_state=0)  # build the model
gbrt.fit(X_train, y_train)

print(gbrt.score(X_train, y_train),gbrt.score(X_test, y_test))
y_pred = gbrt.predict(X_Val)
```

    1.0 0.9740987983978638
    


```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
confmat = confusion_matrix(y_true=y_Val, y_pred=y_pred)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()
print(classification_report(y_Val, y_pred))
```


![png](ML_Final_IndivProject_Racela_1_files/ML_Final_IndivProject_Racela_1_53_0.png)


                  precision    recall  f1-score   support
    
               0       0.98      0.99      0.99      4884
               1       0.28      0.10      0.15       109
    
        accuracy                           0.97      4993
       macro avg       0.63      0.55      0.57      4993
    weighted avg       0.96      0.97      0.97      4993
    
    

Use resample on the train only.


```python
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
X_train, X_test, y_train, y_test = train_test_split(X_t,y_t, stratify=y_t,test_size=0.25, random_state=24)

y_i = y_train
x_i = X_train
X_res, y_res = SMOTE().fit_resample(x_i, y_i)

gbrt = GradientBoostingClassifier(max_depth=9, learning_rate=0.1, random_state=0)  # build the model
gbrt.fit(X_res, y_res)

print(gbrt.score(X_train, y_train),gbrt.score(X_test, y_test))
y_pred = gbrt.predict(X_Val)
```

    1.0 0.9578104138851803
    


```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
confmat = confusion_matrix(y_true=y_Val, y_pred=y_pred)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()
print(classification_report(y_Val, y_pred))
```


![png](/images/brankruptcy_adaboost/brankruptcy_adaboost_56_0.png)


                  precision    recall  f1-score   support
    
               0       0.98      0.98      0.98      4884
               1       0.20      0.28      0.23       109
    
        accuracy                           0.96      4993
       macro avg       0.59      0.63      0.60      4993
    weighted avg       0.97      0.96      0.96      4993
    
    

With the validation set, the f1-score is low for both iterations. Comparing the two, the f1-score has improved on the one that used SMOTE. This indicates that SMOTE can reduce the bias on the majority class and somehow provide a more precise prediction.

## Learning Points

With this project, I have learned a few points. We must be able to take care of imbalanced dataset. There are many dataset in the world that has this characteristics such as Fraud Detection. Being able to take care can make your model resilient to inherent bias. This can be further shown on the validation test conducted. Also, there are many method that can be used within the dataset. Not all method are created equal but can be compatible to certain dataset. Being able to know various model can make your model more accurate. I have been introduced immensely on the boosting algorithm like the Adaptive Boost. These boosting algorithm can convert weak learners to strong ones like a decision tree. Other weak learners can be checked and see if it can improve the models. Though I havent implemented it, I have been also introduced to XGBoost and LightGBM. These boosting algorithm can be the next steps of this report.


