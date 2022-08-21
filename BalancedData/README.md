## **Balanced Data**

*The data is highly imbalanced which creates preoblem for machine learning algorithms to train data with few instances of one target class and two many instances of other target class.*

*To solve this problem of under representation of other classes we need to balance the data in order to make Machine Learning algorithms gernalize well. In this project we use `imblearn` library to balance the data.*

*To balance the data either we need to reduce the size of majority class or increase the size of minority class using synthetic data.*

### **Methods of sampling**

1. Undersmapling
2. Oversampling

### **Importing basic libraries**


```python
import imblearn
import pandas as pd
import numpy
```

### **Importing train set** 


```python
train = pd.read_pickle("data/train.pickle")
```

### **Undersample**

*Under this method we dicard instances of majority class so that our minority class hve significant number of instances in comparison to majority class. We use just 1% of majority class in undersampling*


```python
from imblearn.under_sampling import RandomUnderSampler
under = RandomUnderSampler(sampling_strategy=0.1, random_state=123)
x, y = under.fit_resample(train.iloc[:,:-1], train["Class"])
train_under = pd.concat([x,y], axis=1)
train_under.to_pickle("data/train_under.pickle")
```

### **Oversampling**

*In oversampling synthetic data is generated in order to balance the data set. In our project we have implemented four methods of oversampling*

1. Bootstrap oversampling
2. Oversampling with shrinkage
3. SMOTE
4. ADASYN

#### **Bootstrap Sampling**

*Using bootstrap sampling method we repeat the samples from the minority class in order to balance the data* 


```python
from imblearn.over_sampling import RandomOverSampler
over = RandomOverSampler(sampling_strategy=0.3, random_state=123)
x, y = over.fit_resample(train.iloc[:,:-1], train["Class"])
train_over = pd.concat([x,y], axis=1)
train_over.to_pickle("data/train_over.pickle")
```

#### **Oversampling with shrinkage**

*Using shrinkage we generate a cluster of synthetic data around the already existing minority 
class to generate a smoothed bootstrap instead*


```python
over_shrink = RandomOverSampler(shrinkage=0.3, sampling_strategy=0.3, random_state=123)
x,y = over_shrink.fit_resample(train.iloc[:,:-1], train["Class"])
train_over_shrink = pd.concat([x,y], axis=1)
train_over_shrink.to_pickle("data/train_over_shrink.pickle")
```

#### **SMOTE**

*SMOTE stands for Synthetic Minority Over Sampling Technique. Under this method the minority 
instances connects themselve along a straight line with other instances from same minority class and 
generate synthetic data along these lines* 


```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.3, random_state=123)
x,y = smote.fit_resample(train.iloc[:,:-1], train["Class"])
train_smote = pd.concat([x,y], axis=1)
train_smote.to_pickle("data/train_smote.pickle")
```

#### **ADASYN**

*ADASYN stands for ADaptive SYNthetic algorithm. This method works in a similar 
way as that of SMOTE but it generates data using the total distribution of minority class* 


```python
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(sampling_strategy=0.3, random_state=123)
x,y = adasyn.fit_resample(train.iloc[:,:-1], train["Class"])
train_adasyn = pd.concat([x,y], axis=1)
train_adasyn.to_pickle("data/train_adasyn.pickle")
```


```python

```
