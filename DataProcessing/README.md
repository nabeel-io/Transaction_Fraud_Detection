## **Data Processing**

TABLE OF CONTENTS

1. Importing essential libraries
2. Importing Function
3. Importing data from csv
4. Function to convert data type
5. Conversion to different file formats
6. Memory Usage
7. Test file conversion

### **Importing essential libraries**

*Importing essential libraries like os, zipfile,pandas aslo pyarrow (for feather and parquet)*


```python
import os
import zipfile
import pandas as pd
```

### **Import Function**


```python
# function for importing data
def import_data():
    os.mkdir("data")
    !kaggle competitions download -c santander-customer-transaction-prediction -p data/
    with zipfile.ZipFile("data/santander-customer-transaction-prediction.zip", "r") as zipdata:
        zipdata.extractall("data/")
```


```python
import_data()
```

    Warning: Your Kaggle API key is readable by other users on this system! To fix this, you can run 'chmod 600 /home/nabeel/.kaggle/kaggle.json'
    Downloading santander-customer-transaction-prediction.zip to data
    100%|███████████████████████████████████████▉| 250M/250M [00:52<00:00, 5.67MB/s]
    100%|████████████████████████████████████████| 250M/250M [00:52<00:00, 4.98MB/s]


### **Importing data from csv**

*As we can observe the train dataframe takes 3880 miliseconds. The data type for integer value is in int64 and continuous values
is in float64 format*


```python
%%time
df_train = pd.read_csv("data/train.csv", index_col=False)
```

    CPU times: user 3.69 s, sys: 112 ms, total: 3.8 s
    Wall time: 3.88 s



```python
df_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200000 entries, 0 to 199999
    Columns: 202 entries, ID_code to var_199
    dtypes: float64(200), int64(1), object(1)
    memory usage: 308.2+ MB



```python
df_train.dtypes
```




    ID_code     object
    target       int64
    var_0      float64
    var_1      float64
    var_2      float64
                ...   
    var_195    float64
    var_196    float64
    var_197    float64
    var_198    float64
    var_199    float64
    Length: 202, dtype: object



### **Function to convert data type**

*This function converts the int64 into int16 and float64 to float32 format without disturbing the precision, it also drops
unneccesary columns in order to preserve the memory*


```python
# changing the datatype to save memory space
def d2d(df, drop):
    df = df.drop(columns=drop)
    dict_dtypes = dict(df.dtypes) 
    for col,dtype in dict_dtypes.items():
        if dtype == "int64":
            df[col] = df[col].astype("int16")
        elif dtype == "float64":
            df[col] = df[col].astype("float32")
        else:
            pass
    return df
```

*It took 3420 milliseconds to convert the entire dataset to the given format with almost cutting  the memory requirement 
in half from initial state*


```python
%%time
df_train = d2d(df_train, ['ID_code'])
```

    CPU times: user 6.48 s, sys: 26.4 s, total: 32.9 s
    Wall time: 34.2 s



```python
df_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200000 entries, 0 to 199999
    Columns: 201 entries, target to var_199
    dtypes: float32(200), int16(1)
    memory usage: 153.0 MB



```python
df_train.dtypes
```




    target       int16
    var_0      float32
    var_1      float32
    var_2      float32
    var_3      float32
                ...   
    var_195    float32
    var_196    float32
    var_197    float32
    var_198    float32
    var_199    float32
    Length: 201, dtype: object



### **Conversion to different file formats** 

*We convert our processed data to three different file formats to compare there conversion speed and there read speed in order 
to get the file format which is faster to read and write*


```python
print('Reading and Writing CSV')
%time df_train.to_pickle("data/train.pickle")
%time df_train = pd.read_pickle("data/train.pickle")
```

    Reading and Writing CSV
    CPU times: user 3.58 ms, sys: 130 ms, total: 134 ms
    Wall time: 157 ms
    CPU times: user 5.2 ms, sys: 29.2 ms, total: 34.4 ms
    Wall time: 34 ms



```python
print('Reading and Writing Feather')
%time df_train.to_feather("data/train.feather")
%time df_train = pd.read_feather("data/train.feather")
```

    Reading and Writing Feather
    CPU times: user 724 ms, sys: 185 ms, total: 909 ms
    Wall time: 1.05 s
    CPU times: user 477 ms, sys: 248 ms, total: 725 ms
    Wall time: 1.07 s



```python
print('Reading and Writing Parquet')
%time df_train.to_parquet("data/train.parquet")
%time df_train = pd.read_parquet("data/train.parquet")
```

    Reading and Writing Parquet
    CPU times: user 2.65 s, sys: 136 ms, total: 2.78 s
    Wall time: 2.67 s
    CPU times: user 552 ms, sys: 394 ms, total: 946 ms
    Wall time: 5.38 s


*Out of pickle, feather and parquet file format we find pickle to have the fastest read and and write speed and 
we will prefer to use this format to import data for rest of the project work*

### **Memory Usage**

*Out of the three file formats pickle and feather use the least memory but since the speed of read and write of pickle file 
is much faster than other file format in our case we will prefer to use it*


```python
!ls -GFlash data/train.csv data/train.pickle data/train.feather data/train.parquet
```

    289M -rw-rw-r-- 1 nabeel 289M Jun 12 16:14 data/train.csv
    153M -rw-rw-r-- 1 nabeel 153M Jun 12 16:15 data/train.feather
    156M -rw-rw-r-- 1 nabeel 156M Jun 12 16:15 data/train.parquet
    153M -rw-rw-r-- 1 nabeel 153M Jun 12 16:15 data/train.pickle


### **Test file conversion**


```python
df_test = pd.read_csv("data/test.csv")
df_test = d2d(df_test, ["ID_code"])
df_test.to_pickle("data/test.pickle")
```


```python

```
