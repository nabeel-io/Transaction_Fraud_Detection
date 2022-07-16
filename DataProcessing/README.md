## **Data Processing**

### **Table of Contents**
1. Importing libraries
2. Importing function
3. Importing Dataframe
4. Reducing Memory Usage
5. Target Class proportion
6. Train/Test split
7. Conversion to different file formats
8. Memory Usage
9. Train Test conversion

### **Importing libraries**
*Importing essential libraries like `os` for making directory, `zipfile` for extracting data from zip folder, `pandas` to process
the dataframe and `pyarrow` for feather and parquet*


```python
import os
import zipfile
import pandas as pd
```

### **Import Function**
*Importing data from `Kaggle API`*


```python
def import_data():
    if not os.path.exists("customer_transaction_prediction/"):
        os.makedirs("data")
        !kaggle datasets download -d mlg-ulb/creditcardfraud -p data/
        with zipfile.ZipFile("data/creditcardfraud.zip", "r") as zipdata:
            zipdata.extractall("data/")
```


```python
import_data()
```

    Warning: Your Kaggle API key is readable by other users on this system! To fix this, you can run 'chmod 600 /home/nabeel/.kaggle/kaggle.json'
    Downloading creditcardfraud.zip to data
    100%|██████████████████████████████████████| 66.0M/66.0M [00:18<00:00, 4.41MB/s]
    100%|██████████████████████████████████████| 66.0M/66.0M [00:18<00:00, 3.64MB/s]


### **Importing Dataframe**
*The dataframe `df` consist of 29 attributes and 1 target class with 284,807 instances using 67.4 Mb of memory without null values*


```python
df = pd.read_csv("data/creditcard.csv")
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 284807 entries, 0 to 284806
    Data columns (total 31 columns):
     #   Column  Non-Null Count   Dtype  
    ---  ------  --------------   -----  
     0   Time    284807 non-null  float64
     1   V1      284807 non-null  float64
     2   V2      284807 non-null  float64
     3   V3      284807 non-null  float64
     4   V4      284807 non-null  float64
     5   V5      284807 non-null  float64
     6   V6      284807 non-null  float64
     7   V7      284807 non-null  float64
     8   V8      284807 non-null  float64
     9   V9      284807 non-null  float64
     10  V10     284807 non-null  float64
     11  V11     284807 non-null  float64
     12  V12     284807 non-null  float64
     13  V13     284807 non-null  float64
     14  V14     284807 non-null  float64
     15  V15     284807 non-null  float64
     16  V16     284807 non-null  float64
     17  V17     284807 non-null  float64
     18  V18     284807 non-null  float64
     19  V19     284807 non-null  float64
     20  V20     284807 non-null  float64
     21  V21     284807 non-null  float64
     22  V22     284807 non-null  float64
     23  V23     284807 non-null  float64
     24  V24     284807 non-null  float64
     25  V25     284807 non-null  float64
     26  V26     284807 non-null  float64
     27  V27     284807 non-null  float64
     28  V28     284807 non-null  float64
     29  Amount  284807 non-null  float64
     30  Class   284807 non-null  int64  
    dtypes: float64(30), int64(1)
    memory usage: 67.4 MB


### **Reducing Memory Usage**
*Reducing the size of the dataset without significantly effecting the precision of sample values. 
Converting the target class to `int16` and column variables to `float32` data type.* 


```python
# reducing the size of dataset without effecting the precision of sample values
def reduce(df):
    dict_dtypes = dict(df.dtypes)
    for col, dtype in dict_dtypes.items():
        if dtype == "float64":
            df[col] = df[col].astype("float32")
        elif dtype == "int64":
            df[col] = df[col].astype("int16")
        else:
            pass
    return df
```


```python
reduced_df = reduce(df)
```

### **Target Class proportion**
*Majority of instances are non fradulent in nature, just 1.7% of instances consist of fradulent payment.*


```python
# find the weight of each class
df["Class"].value_counts()/len(df)
```




    0    0.998273
    1    0.001727
    Name: Class, dtype: float64



### **Train/Test Split**
*Splitting the dataframe into train and test sets using stratified sampling which preserves the proportion of target class.
Keeping 25% of instances for test and rest for training.*


```python
# split the data into training and testing sets using stratified fold
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(reduced_df.iloc[:, :-1], 
                                                reduced_df["Class"],
                                                stratify=reduced_df["Class"],
                                                test_size=0.25,
                                                random_state=123)

```


```python
train = pd.concat([xtrain, ytrain], axis=1)
train = train.reset_index(drop=True)
test = pd.concat([xtest, ytest], axis=1)
test = test.reset_index(drop=True)
```


```python
# target class proportions are preserved
print("train set proportions: ")
print(train["Class"].value_counts()/len(train))
```

    train set proportions: 
    0    0.998273
    1    0.001727
    Name: Class, dtype: float64


### **Conversion to different file formats**
*We convert our processed data to three different file formats to compare there conversion speed and there read speed in order 
to get the file format which is faster to read and write*


```python
print("Reading and Writing in Pickle")
%time df.to_pickle("data/df.pickle")
%time df = pd.read_pickle("data/df.pickle")
```

    Reading and Writing in Pickle
    CPU times: user 0 ns, sys: 47.1 ms, total: 47.1 ms
    Wall time: 45.9 ms
    CPU times: user 2.43 ms, sys: 8.59 ms, total: 11 ms
    Wall time: 9.92 ms



```python
print("Reading and Writing in Feather")
%time df.to_feather("data/df.feather")
%time df = pd.read_feather("data/df.feather")
```

    Reading and Writing in Feather
    CPU times: user 184 ms, sys: 39.8 ms, total: 224 ms
    Wall time: 75.7 ms
    CPU times: user 57.3 ms, sys: 52.7 ms, total: 110 ms
    Wall time: 25.8 ms



```python
print("Reading and Writing in Parquet")
%time df.to_parquet("data/df.parquet")
%time df = pd.read_parquet("data/df.parquet")
```

    Reading and Writing in Parquet
    CPU times: user 817 ms, sys: 45.1 ms, total: 862 ms
    Wall time: 814 ms
    CPU times: user 202 ms, sys: 93.7 ms, total: 295 ms
    Wall time: 125 ms


*Out of pickle, feather and parquet file format we find pickle to have the fastest read and and write speed and 
we will prefer to use this format to import data for rest of the project work*

### **Memory usage**
*Out of the three file formats pickle and feather use the least memory, almost reducing the size of data four times than that of
orignal csv file.We prefer to use pickle in our case becuase it is much faster in read and write speeds than other file formats*


```python
!ls -GFlash data/creditcard.csv data/df.pickle data/df.feather data/df.parquet
```

    144M -rw-rw-r-- 1 nabeel 144M Jul 16 21:07 data/creditcard.csv
     32M -rw-rw-r-- 1 nabeel  32M Jul 16 21:07 data/df.feather
     49M -rw-rw-r-- 1 nabeel  49M Jul 16 21:07 data/df.parquet
     34M -rw-rw-r-- 1 nabeel  34M Jul 16 21:07 data/df.pickle


### **Train Test conversion**


```python
train.to_pickle("data/train.pickle")
test.to_pickle("data/test.pickle")
```


```python

```


```python

```
