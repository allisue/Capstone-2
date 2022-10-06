```python
# Predicting Hospital Patient Length of Stay
```


```python

```


```python
### Since the beginning of the COVID-19 pandemic, hospitals have consistently struggled with patient bed, resource, and staffing shortages, and being able to forecast patient stay length will help hospitals to predict bed/room availability, age-appropriate care, staffing, and resource needs, as well as ensure financial security.
```


```python

```


```python
## Issues
```


```python
### Issue: Ongoing financial instability is a problem across America's hospital as they recover from the height of the COVID-19 pandemic.
```


```python
### Solution: Hospitals will be better able to predict revenue to ensure financial security through patient Length of Stay prediction.
```


```python

```


```python
### Issue: Age is a determinant of educational and counseling strategies as well as recovery outcome, and hospitals cannot prepare for this without a forecast of age care needs. 
```


```python
### Solution: Hospitals will be better able to forecast needs for age appropriate care through patient Length of Stay prediction.
```


```python

```


```python
### Issue: The nurse shortage in America is only increasing, and miscommunication in staffing needs is a contributor.
```


```python
### Solution: Hospitals will be better able to forecast staffing needs through patient Length of Stay prediction.
```


```python

```


```python
### Issue: The COVID-19 pandemic shined a light on how the healthcare system struggles when faced with a lack of patient beds and resources. 
```


```python
### Solution: Hospitals will be better able to forecast bed/resource needs through patient Length of Stay prediction.
```


```python

```


```python
## Importing extensions
```


```python
import numpy as np
```


```python
import pandas as pd
```


```python
import matplotlib.pyplot as plt
```


```python
import seaborn as sns
```


```python
import statsmodels.api as sm
sns.set_style("darkgrid")
```


```python
## Uploading tables 
```


```python
# <span style="color:blue">Descriptive Statistics</span>
```


```python
## Train table
```


```python
train = pd.read_csv("train_data.csv")
#train.info()
#train.describe()
```


```python
train.head()
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
      <th>case_id</th>
      <th>Hospital_code</th>
      <th>Hospital_type_code</th>
      <th>City_Code_Hospital</th>
      <th>Hospital_region_code</th>
      <th>Available Extra Rooms in Hospital</th>
      <th>Department</th>
      <th>Ward_Type</th>
      <th>Ward_Facility_Code</th>
      <th>Bed Grade</th>
      <th>patientid</th>
      <th>City_Code_Patient</th>
      <th>Type of Admission</th>
      <th>Severity of Illness</th>
      <th>Visitors with Patient</th>
      <th>Age</th>
      <th>Admission_Deposit</th>
      <th>Stay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>8</td>
      <td>c</td>
      <td>3</td>
      <td>Z</td>
      <td>3</td>
      <td>radiotherapy</td>
      <td>R</td>
      <td>F</td>
      <td>2.0</td>
      <td>31397</td>
      <td>7.0</td>
      <td>Emergency</td>
      <td>Extreme</td>
      <td>2</td>
      <td>51-60</td>
      <td>4911.0</td>
      <td>0-10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>c</td>
      <td>5</td>
      <td>Z</td>
      <td>2</td>
      <td>radiotherapy</td>
      <td>S</td>
      <td>F</td>
      <td>2.0</td>
      <td>31397</td>
      <td>7.0</td>
      <td>Trauma</td>
      <td>Extreme</td>
      <td>2</td>
      <td>51-60</td>
      <td>5954.0</td>
      <td>41-50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>10</td>
      <td>e</td>
      <td>1</td>
      <td>X</td>
      <td>2</td>
      <td>anesthesia</td>
      <td>S</td>
      <td>E</td>
      <td>2.0</td>
      <td>31397</td>
      <td>7.0</td>
      <td>Trauma</td>
      <td>Extreme</td>
      <td>2</td>
      <td>51-60</td>
      <td>4745.0</td>
      <td>31-40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>26</td>
      <td>b</td>
      <td>2</td>
      <td>Y</td>
      <td>2</td>
      <td>radiotherapy</td>
      <td>R</td>
      <td>D</td>
      <td>2.0</td>
      <td>31397</td>
      <td>7.0</td>
      <td>Trauma</td>
      <td>Extreme</td>
      <td>2</td>
      <td>51-60</td>
      <td>7272.0</td>
      <td>41-50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>26</td>
      <td>b</td>
      <td>2</td>
      <td>Y</td>
      <td>2</td>
      <td>radiotherapy</td>
      <td>S</td>
      <td>D</td>
      <td>2.0</td>
      <td>31397</td>
      <td>7.0</td>
      <td>Trauma</td>
      <td>Extreme</td>
      <td>2</td>
      <td>51-60</td>
      <td>5558.0</td>
      <td>41-50</td>
    </tr>
  </tbody>
</table>
</div>




```python
#print(train.isnull().sum())
```


```python
### Dropping columns with null values 
```


```python
print(train.shape)
```

    (318438, 18)
    


```python
train.drop("Bed Grade", axis= "columns", inplace=True)
```


```python
print(train.shape)
```

    (318438, 17)
    


```python
train.drop("City_Code_Patient", axis= "columns", inplace=True)
```


```python
print(train.shape)
```

    (318438, 16)
    


```python
### Identifying data types of each column to check compatibility
```


```python
train.dtypes
```




    case_id                                int64
    Hospital_code                          int64
    Hospital_type_code                    object
    City_Code_Hospital                     int64
    Hospital_region_code                  object
    Available Extra Rooms in Hospital      int64
    Department                            object
    Ward_Type                             object
    Ward_Facility_Code                    object
    patientid                              int64
    Type of Admission                     object
    Severity of Illness                   object
    Visitors with Patient                  int64
    Age                                   object
    Admission_Deposit                    float64
    Stay                                  object
    dtype: object




```python

```


```python
## Looking at Stay Length
```


```python
### Counting the unique values in 'Stay'

Stay_count = train['Stay'].value_counts(sort= True)
Stay_count
```




    21-30                 87491
    11-20                 78139
    31-40                 55159
    51-60                 35018
    0-10                  23604
    41-50                 11743
    71-80                 10254
    More than 100 Days     6683
    81-90                  4838
    91-100                 2765
    61-70                  2744
    Name: Stay, dtype: int64




```python
### Expressing the counts as proportions
print(train.Stay.value_counts(normalize= True))
```

    21-30                 0.274751
    11-20                 0.245382
    31-40                 0.173217
    51-60                 0.109968
    0-10                  0.074124
    41-50                 0.036877
    71-80                 0.032201
    More than 100 Days    0.020987
    81-90                 0.015193
    91-100                0.008683
    61-70                 0.008617
    Name: Stay, dtype: float64
    


```python
### Visualizing counts
sns.set_style("darkgrid")

category_order = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', 'More than 100 Days']

sns.countplot(y= "Stay", data= train, order= category_order)

plt.title('Stay Length of Patients (Days)')
```




    Text(0.5, 1.0, 'Stay Length of Patients (Days)')




    
![png](output_41_1.png)
    



```python
## INSIGHTS:
⮞ Stay Lengths of 11-20 and 21-30 Days are most common, while Stay Lengths suddenly increase at 51-60 Days
⮞ 11-20 Days: 78,139 patients, an increase of 🡹 54,525 compared to 0-10 Days
⮞ 21-30 Days: 87,491 patients, an increase of 🡹 9,352 compared to 11-20 Days
⮞ 51-60 Days: 35,018 patients, an increase of 🡹 23,275 compared to 41-50 Days, and 🡹 32,274 compared to 61-70 Days
```


      Input In [482]
        ⮞ Stay Lengths of 11-20 and 21-30 Days are most common, while Stay Lengths suddenly increase at 51-60 Days
        ^
    SyntaxError: invalid character '⮞' (U+2B9E)
    



```python

```


```python
## Looking at Ages by Stay Length
```


```python
### Counting the unique values in 'Age'

Stay_count = train['Stay'].value_counts(sort= True)
Stay_count
```




    21-30                 87491
    11-20                 78139
    31-40                 55159
    51-60                 35018
    0-10                  23604
    41-50                 11743
    71-80                 10254
    More than 100 Days     6683
    81-90                  4838
    91-100                 2765
    61-70                  2744
    Name: Stay, dtype: int64




```python
### Expressing the counts as proportions
print(train.Stay.value_counts(normalize= True))
```

    21-30                 0.274751
    11-20                 0.245382
    31-40                 0.173217
    51-60                 0.109968
    0-10                  0.074124
    41-50                 0.036877
    71-80                 0.032201
    More than 100 Days    0.020987
    81-90                 0.015193
    91-100                0.008683
    61-70                 0.008617
    Name: Stay, dtype: float64
    


```python
### Age distribution for each stay length
stay_age_distribution = pd.get_dummies(train.loc[:,['Stay','Age']], columns=['Age']).groupby('Stay', as_index=False).sum()
stay_age_distribution
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
      <th>Stay</th>
      <th>Age_0-10</th>
      <th>Age_11-20</th>
      <th>Age_21-30</th>
      <th>Age_31-40</th>
      <th>Age_41-50</th>
      <th>Age_51-60</th>
      <th>Age_61-70</th>
      <th>Age_71-80</th>
      <th>Age_81-90</th>
      <th>Age_91-100</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0-10</td>
      <td>615.0</td>
      <td>1552.0</td>
      <td>3467.0</td>
      <td>4916.0</td>
      <td>4727.0</td>
      <td>3427.0</td>
      <td>2194.0</td>
      <td>2201.0</td>
      <td>422.0</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11-20</td>
      <td>1959.0</td>
      <td>5343.0</td>
      <td>11272.0</td>
      <td>15792.0</td>
      <td>14959.0</td>
      <td>11346.0</td>
      <td>7870.0</td>
      <td>7958.0</td>
      <td>1392.0</td>
      <td>248.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21-30</td>
      <td>1489.0</td>
      <td>4312.0</td>
      <td>11394.0</td>
      <td>18550.0</td>
      <td>17906.0</td>
      <td>13058.0</td>
      <td>9033.0</td>
      <td>9534.0</td>
      <td>1920.0</td>
      <td>295.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31-40</td>
      <td>1014.0</td>
      <td>2681.0</td>
      <td>6912.0</td>
      <td>10912.0</td>
      <td>10983.0</td>
      <td>8569.0</td>
      <td>5930.0</td>
      <td>6420.0</td>
      <td>1504.0</td>
      <td>234.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41-50</td>
      <td>187.0</td>
      <td>510.0</td>
      <td>1398.0</td>
      <td>2373.0</td>
      <td>2507.0</td>
      <td>1735.0</td>
      <td>1205.0</td>
      <td>1383.0</td>
      <td>379.0</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>51-60</td>
      <td>582.0</td>
      <td>1429.0</td>
      <td>3793.0</td>
      <td>6517.0</td>
      <td>7189.0</td>
      <td>5739.0</td>
      <td>4081.0</td>
      <td>4433.0</td>
      <td>1082.0</td>
      <td>173.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>61-70</td>
      <td>26.0</td>
      <td>89.0</td>
      <td>263.0</td>
      <td>509.0</td>
      <td>562.0</td>
      <td>448.0</td>
      <td>325.0</td>
      <td>378.0</td>
      <td>115.0</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>71-80</td>
      <td>153.0</td>
      <td>350.0</td>
      <td>1026.0</td>
      <td>1807.0</td>
      <td>2146.0</td>
      <td>1710.0</td>
      <td>1230.0</td>
      <td>1367.0</td>
      <td>402.0</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>81-90</td>
      <td>84.0</td>
      <td>223.0</td>
      <td>546.0</td>
      <td>801.0</td>
      <td>885.0</td>
      <td>784.0</td>
      <td>600.0</td>
      <td>670.0</td>
      <td>216.0</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>91-100</td>
      <td>35.0</td>
      <td>71.0</td>
      <td>231.0</td>
      <td>484.0</td>
      <td>578.0</td>
      <td>499.0</td>
      <td>330.0</td>
      <td>386.0</td>
      <td>132.0</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>More than 100 Days</td>
      <td>110.0</td>
      <td>208.0</td>
      <td>541.0</td>
      <td>978.0</td>
      <td>1307.0</td>
      <td>1199.0</td>
      <td>889.0</td>
      <td>1062.0</td>
      <td>326.0</td>
      <td>63.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
### Creating a For loop to create subplots for all age groups
Ages = stay_age_distribution.columns.tolist()
Ages.remove('Stay')
print(Ages)

plt.figure(figsize=(15,20))
plt.suptitle("Subplots for each Age Group", fontsize=20)
x = 1
for age in Ages:
    ax = plt.subplot(int(len(Ages)//2),2,x)
    ax = plt.subplots_adjust(wspace=0.5,hspace=0.4)
    sns.barplot(y = 'Stay', x = age, data = stay_age_distribution, ax = ax)
    plt.title(f'Stay Length by Age Group -> {age}')
    x +=1
```

    ['Age_0-10', 'Age_11-20', 'Age_21-30', 'Age_31-40', 'Age_41-50', 'Age_51-60', 'Age_61-70', 'Age_71-80', 'Age_81-90', 'Age_91-100']
    


    
![png](output_48_1.png)
    



```python
## INSIGHTS
⮞ Majority of Admissions: Age Groups 31-40 (40,859), 41-50 (40,054), 51-60 (30,143)
⮞ 11-20 Days: Age Groups 11-20 🡹, 51-60 🡹, and 61-70 🡹 from previous groups
⮞ 21-30 Days: Age Groups 0-10 🡻, 11-20 🡻, and 81-90 🡹 from previous groups
⮞ 51-60 Days: Age Groups 41-50 🡹, 51-60 🡹, and 61-70 🡹 from previous groups
⮞ All Age Groups increase at Stay Length of 51-60 Days, which reflects the increase when looking at overall Stay Lengths, with Age Group 81-90 the most responsible 
```


      Input In [488]
        ⮞ Majority of Admissions: Age Groups 31-40 (40,859), 41-50 (40,054), 51-60 (30,143)
        ^
    SyntaxError: invalid character '⮞' (U+2B9E)
    



```python

```


```python
## Looking at Departments by Stay Length
```


```python
### Counting the unique values in 'Department'
Dept_count = train['Department'].value_counts(sort= True)
Dept_count
```




    gynecology            249486
    anesthesia             29649
    radiotherapy           28516
    TB & Chest disease      9586
    surgery                 1201
    Name: Department, dtype: int64




```python
### Expressing the counts as proportions
Dept_count = train['Department'].value_counts(normalize= True)
Dept_count
```




    gynecology            0.783468
    anesthesia            0.093108
    radiotherapy          0.089550
    TB & Chest disease    0.030103
    surgery               0.003772
    Name: Department, dtype: float64




```python
### Department distribution for each stay length
stay_dept_distribution = pd.get_dummies(train.loc[:,['Stay','Department']], columns=['Department']).groupby('Stay', as_index=False).sum()
stay_dept_distribution
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
      <th>Stay</th>
      <th>Department_TB &amp; Chest disease</th>
      <th>Department_anesthesia</th>
      <th>Department_gynecology</th>
      <th>Department_radiotherapy</th>
      <th>Department_surgery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0-10</td>
      <td>791.0</td>
      <td>1519.0</td>
      <td>18849.0</td>
      <td>2379.0</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11-20</td>
      <td>2379.0</td>
      <td>8823.0</td>
      <td>59690.0</td>
      <td>6988.0</td>
      <td>259.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21-30</td>
      <td>2618.0</td>
      <td>8950.0</td>
      <td>68383.0</td>
      <td>7263.0</td>
      <td>277.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31-40</td>
      <td>1753.0</td>
      <td>4958.0</td>
      <td>43535.0</td>
      <td>4713.0</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41-50</td>
      <td>300.0</td>
      <td>851.0</td>
      <td>9671.0</td>
      <td>879.0</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>51-60</td>
      <td>1047.0</td>
      <td>2527.0</td>
      <td>27989.0</td>
      <td>3256.0</td>
      <td>199.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>61-70</td>
      <td>76.0</td>
      <td>173.0</td>
      <td>2226.0</td>
      <td>264.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>71-80</td>
      <td>294.0</td>
      <td>703.0</td>
      <td>8165.0</td>
      <td>1043.0</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>81-90</td>
      <td>140.0</td>
      <td>384.0</td>
      <td>3748.0</td>
      <td>528.0</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>91-100</td>
      <td>64.0</td>
      <td>156.0</td>
      <td>2269.0</td>
      <td>256.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>More than 100 Days</td>
      <td>124.0</td>
      <td>605.0</td>
      <td>4961.0</td>
      <td>947.0</td>
      <td>46.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
### Visualizing the Departments by Stay Lengths

Stay_index = train.Stay.value_counts().index[:11]
Stay_index

Department_codeindex= train.Department.value_counts().index[:6]
Department_codeindex

subdata = train[(train.Department.isin(Department_codeindex)) & (train.Stay.isin(Stay_index))]

cf = pd.crosstab(columns=subdata.Stay, index = subdata.Department) 
cf

plt.figure()
Dept_Stay = pd.crosstab(columns=subdata.Stay, index = subdata.Department) 
cf.plot.bar(figsize = (20,8), )
plt.legend(loc = 'best')
plt. savefig('100dpi3.png', dpi=100)
plt.title("Stays Lengths by Department")
plt.ylabel('Patients')
plt.show()
```


    <Figure size 432x288 with 0 Axes>



    
![png](output_55_1.png)
    



```python
## INSIGHTS:
⮞ Gynecology has the most admitted patients (249,486), followed by anesthesia (29,649), radiotherapy (28,516),
TB & Chest disease (9,586), surgery (1,201)
⮞ The most common Stay Length among all Departments is 21-30 Days, followed by 11-20 Days
⮞ Surgery has the largest ratio of patients who stay more than 100 Days
```


      Input In [493]
        ⮞ Gynecology has the most admitted patients (249,486), followed by anesthesia (29,649), radiotherapy (28,516),
        ^
    SyntaxError: invalid character '⮞' (U+2B9E)
    



```python

```


```python
## Severity of Illness by Stay Length
```


```python
### Counting unique values in "Severity of Illness"
SIL_count = train['Severity of Illness'].value_counts(sort= True)
SIL_count
```




    Moderate    175843
    Minor        85872
    Extreme      56723
    Name: Severity of Illness, dtype: int64




```python
### Expressing the counts as proportions
print(train['Severity of Illness'].value_counts(normalize= True))
```

    Moderate    0.552205
    Minor       0.269666
    Extreme     0.178129
    Name: Severity of Illness, dtype: float64
    


```python
### Severity of Illness distribution for each stay length
stay_SIL_distribution = pd.get_dummies(train.loc[:,['Stay','Severity of Illness']], columns= ['Severity of Illness']).groupby('Stay', as_index=False).sum()
stay_SIL_distribution
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
      <th>Stay</th>
      <th>Severity of Illness_Extreme</th>
      <th>Severity of Illness_Minor</th>
      <th>Severity of Illness_Moderate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0-10</td>
      <td>3399.0</td>
      <td>7866.0</td>
      <td>12339.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11-20</td>
      <td>10518.0</td>
      <td>27081.0</td>
      <td>40540.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21-30</td>
      <td>15502.0</td>
      <td>21535.0</td>
      <td>50454.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31-40</td>
      <td>10086.0</td>
      <td>14447.0</td>
      <td>30626.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41-50</td>
      <td>2351.0</td>
      <td>3000.0</td>
      <td>6392.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>51-60</td>
      <td>7777.0</td>
      <td>7128.0</td>
      <td>20113.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>61-70</td>
      <td>647.0</td>
      <td>519.0</td>
      <td>1578.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>71-80</td>
      <td>2575.0</td>
      <td>1928.0</td>
      <td>5751.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>81-90</td>
      <td>1113.0</td>
      <td>985.0</td>
      <td>2740.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>91-100</td>
      <td>805.0</td>
      <td>425.0</td>
      <td>1535.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>More than 100 Days</td>
      <td>1950.0</td>
      <td>958.0</td>
      <td>3775.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
### Visualizing Stay Length for each Severity of Illness
Stay_index = train.Stay.value_counts().index[:11]
Stay_index

SIL_index= train['Severity of Illness'].value_counts().index[:7]
SIL_index

subdata = train[(train['Severity of Illness'].isin(SIL_index)) & (train.Stay.isin(Stay_index))]

cf = pd.crosstab(columns=subdata.Stay, index = subdata['Severity of Illness']) 
cf

plt.figure()
SIL_Stay = pd.crosstab(columns=subdata.Stay, index = subdata['Severity of Illness']) 
cf.plot.bar(figsize = (20,8), )
plt.legend(loc = 'best')
plt. savefig('100dpi3.png', dpi=100)
plt.title("Stays Lengths by Severity of Illness")
plt.ylabel('Patients')
plt.show()
```


    <Figure size 432x288 with 0 Axes>



    
![png](output_62_1.png)
    



```python
### INSIGHTS:
⮞The majority of patients are identified as Moderately ill (175,843 patients), followed by Minorly ill (85,872 patients), and finally Extreme illness (56,723 patients)
⮞ There does not seem to be a pattern to Stay Length in the patients identified as Moderately ill
⮞ Patients identified as Minorly ill seem to stay for a shorter amount of time, with a larger amount staying 0-10 Days, and smaller amount staying More than 100 Days
⮞  Patients identified as Extremely ill seem to stay for a longer amount of time, with a smaller amount staying 0-10 Days, and a larger amount staying More than 100 Days
```


      Input In [498]
        ⮞The majority of patients are identified as Moderately ill (175,843 patients), followed by Minorly ill (85,872 patients), and finally Extreme illness (56,723 patients)
        ^
    SyntaxError: invalid character '⮞' (U+2B9E)
    



```python

```


```python
## Type of Admission by Stay Length
```


```python
### Counting unique values in "Type of Admission"
TOA_count = train['Type of Admission'].value_counts(sort= True)
TOA_count
```




    Trauma       152261
    Emergency    117676
    Urgent        48501
    Name: Type of Admission, dtype: int64




```python
### Expressing the counts as proportions
print(train['Type of Admission'].value_counts(normalize= True))
```

    Trauma       0.478150
    Emergency    0.369541
    Urgent       0.152309
    Name: Type of Admission, dtype: float64
    


```python
# TOA distribution for each stay length
stay_TOA_distribution = pd.get_dummies(train.loc[:,['Stay','Type of Admission']], columns=['Type of Admission']).groupby('Stay', as_index=False).sum()
stay_TOA_distribution
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
      <th>Stay</th>
      <th>Type of Admission_Emergency</th>
      <th>Type of Admission_Trauma</th>
      <th>Type of Admission_Urgent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0-10</td>
      <td>14218.0</td>
      <td>5328.0</td>
      <td>4058.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11-20</td>
      <td>31559.0</td>
      <td>33745.0</td>
      <td>12835.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21-30</td>
      <td>27399.0</td>
      <td>46244.0</td>
      <td>13848.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31-40</td>
      <td>18921.0</td>
      <td>28107.0</td>
      <td>8131.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41-50</td>
      <td>4145.0</td>
      <td>5968.0</td>
      <td>1630.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>51-60</td>
      <td>11885.0</td>
      <td>18310.0</td>
      <td>4823.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>61-70</td>
      <td>976.0</td>
      <td>1479.0</td>
      <td>289.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>71-80</td>
      <td>3375.0</td>
      <td>5551.0</td>
      <td>1328.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>81-90</td>
      <td>1708.0</td>
      <td>2548.0</td>
      <td>582.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>91-100</td>
      <td>935.0</td>
      <td>1518.0</td>
      <td>312.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>More than 100 Days</td>
      <td>2555.0</td>
      <td>3463.0</td>
      <td>665.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
### Visualizing Stay Length for each Type of Admission

Stay_index = train.Stay.value_counts().index[:11]
Stay_index

TOA_index= train['Type of Admission'].value_counts().index[:4]
TOA_index

subdata = train[(train['Type of Admission'].isin(TOA_index)) & (train.Stay.isin(Stay_index))]

cf = pd.crosstab(columns=subdata.Stay, index = subdata['Type of Admission']) 
cf

plt.figure()
TOA_Stay = pd.crosstab(columns=subdata.Stay, index = subdata['Type of Admission']) 
cf.plot.bar(figsize = (20,8), )
plt.legend(loc = 'best')
plt. savefig('100dpi3.png', dpi=100)
plt.title("Stays Lengths by Type of Admission")
plt.ylabel('Patients')
plt.show()
```


    <Figure size 432x288 with 0 Axes>



    
![png](output_69_1.png)
    



```python
### INSIGHTS:
⮞The leading Type of Admission is Trauma (152261,~47.81%), followed by Emergency (117676, ~36.95%), and lastly Urgent (48501, ~15.23%)
⮞ 58958, or ~66% of Emergency, 79989, or ~71% of Trauma, and 26683, or ~72% of Urgent Admissions stay between 11-20 to 21-30 days
⮞ Emergency Admissions are more likely to stay between 0-10 days to 11-20 days than any other group
```


      Input In [504]
        ⮞The leading Type of Admission is Trauma (152261,~47.81%), followed by Emergency (117676, ~36.95%), and lastly Urgent (48501, ~15.23%)
        ^
    SyntaxError: invalid character '⮞' (U+2B9E)
    



```python

```


```python
## Admission Deposit by Stay Length
```


```python
### Identifying the average overall Admission Deposit
AD_Avg = train['Admission_Deposit'].mean()
AD_Avg
```




    4880.749392346391




```python
### Vizualizing Average Admission Deposit for each stay length
category_order = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', 'More than 100 Days']

sns.set_style("darkgrid")
plot = sns.catplot(x= "Stay", y= "Admission_Deposit", data= train, kind= "bar", order= category_order)
rotation = 90
for axis in plot.fig.axes:  
     axis.set_xticklabels(axis.get_xticklabels(), rotation = rotation)
plt.title("Average Admission Deposits by Stay Length")
plt.ylabel('Avg. Admission Deposit ($)')
plt.show()
```


    
![png](output_74_0.png)
    



```python
### INSIGHTS

⮞ There is not much variance in the Admission Deposits (AD) across all Stay Lengths
⮞ The average overall AD was $4880.75
⮞ Stay Lengths of 21-30 days had the highest average AD of ~$5000
⮞ Stay Lengths of 81-90 days had the lowest average AD of ~$4500
⮞ Stay Lengths of 61-70 days had the largest range of data
⮞ After 0-10 days ADs increase overall, and after 41-50 days ADs decrease overall
```


      Input In [508]
        ⮞ There is not much variance in the Admission Deposits (AD) across all Stay Lengths
        ^
    SyntaxError: invalid character '⮞' (U+2B9E)
    



```python

```


```python
# <span style="color:blue">Inferential Statistics</span>
```


```python
## Regression Analysis
```


```python
### Importing Train data that has been converted into Nominal and Ordinal Variables for statistical analysis
stats = pd.read_csv("train_data2.csv")
```


```python
stats.head()
```


```python
### Dropping irrelevant columns
```


```python
stats.drop("Regression formula", axis= "columns", inplace=True)
```


```python
stats.drop("Unnamed: 6", axis= "columns", inplace=True)
```


```python
stats.drop("Unnamed: 7", axis= "columns", inplace=True)
```


```python
stats.head()
```


```python
### Correlation Heatmap to identify variable correlation

columns = ['TOA', 'Department', 'SOI','Age', 'Stay']
corr_df = stats[columns].corr()
df_heatmap = sns.heatmap(corr_df, cmap='YlOrRd', annot=True)
plt.title('Correlation Heatmap')
```




    Text(0.5, 1.0, 'Correlation Heatmap')




    
![png](output_86_1.png)
    



```python
## INSIGHT
⮞ SOI (Severity of Illness) (.09) and TOA (Type of Admission) (0.08) seem to have a slight correlation with Stay Length.
```


      Input In [510]
        ⮞ SOI (Severity of Illness) (.09) and TOA (Type of Admission) (0.08) seem to have a slight correlation with Stay Length.
        ^
    SyntaxError: invalid character '⮞' (U+2B9E)
    



```python

```


```python
### Visualizing Variable Correlation with Coefficient plot

col_list_corr = corr_df.columns
corr_df=corr_df.sort_values('Stay', ascending=False)
corr_df['Stay'].plot(kind='bar')
plt.xlabel('Variables')
plt.ylabel('Coefficient')
plt.title('Variable Correlation with Stay Length')
```




    Text(0.5, 1.0, 'Variable Correlation with Stay Length')




    
![png](output_89_1.png)
    



```python
## INSIGHT
⮞ This confirms that SOI (Severity of Illness) (.09) and TOA (Type of Admission) (.08) seem to have a slight correlation with Stay Length.
```


      Input In [512]
        ⮞ This confirms that SOI (Severity of Illness) (.09) and TOA (Type of Admission) (.08) seem to have a slight correlation with Stay Length.
        ^
    SyntaxError: invalid character '⮞' (U+2B9E)
    



```python

```


```python
### OLS Regression Model

columns_1 = ['TOA', 'Department', 'SOI','Age']
columns_2 = ['Stay']

independent_variables = stats[columns_1]
dependent_variables = stats[columns_2]

independent_variables = sm.add_constant(independent_variables)

regression_model = sm.OLS(dependent_variables,independent_variables).fit()
regression_model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Stay</td>       <th>  R-squared:         </th>  <td>   0.017</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.017</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   1348.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 06 Oct 2022</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>00:42:36</td>     <th>  Log-Likelihood:    </th> <td>-6.3889e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>318438</td>      <th>  AIC:               </th>  <td>1.278e+06</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>318433</td>      <th>  BIC:               </th>  <td>1.278e+06</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>      <td>    1.6689</td> <td>    0.013</td> <td>  123.630</td> <td> 0.000</td> <td>    1.642</td> <td>    1.695</td>
</tr>
<tr>
  <th>TOA</th>        <td>    0.1758</td> <td>    0.004</td> <td>   39.694</td> <td> 0.000</td> <td>    0.167</td> <td>    0.185</td>
</tr>
<tr>
  <th>Department</th> <td>    0.0625</td> <td>    0.005</td> <td>   12.976</td> <td> 0.000</td> <td>    0.053</td> <td>    0.072</td>
</tr>
<tr>
  <th>SOI</th>        <td>    0.2298</td> <td>    0.005</td> <td>   47.393</td> <td> 0.000</td> <td>    0.220</td> <td>    0.239</td>
</tr>
<tr>
  <th>Age</th>        <td>    0.0463</td> <td>    0.002</td> <td>   26.887</td> <td> 0.000</td> <td>    0.043</td> <td>    0.050</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>42108.884</td> <th>  Durbin-Watson:     </th> <td>   1.694</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>61192.341</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.009</td>   <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 3.734</td>   <th>  Cond. No.          </th> <td>    22.6</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
### INSIGHTS
⮞  R Squared (~.02%), suggesting this model is a poor fit. 
```


      Input In [514]
        ⮞  R Squared (~.02%), suggesting this model is a poor fit.
        ^
    SyntaxError: invalid character '⮞' (U+2B9E)
    



```python

```


```python
## Analysis and Vizualization of Coefficients

columns_1 = ['TOA', 'Department', 'SOI','Age']
columns_2 = ['Stay']

independent_variables = stats[columns_1]
dependent_variables = stats[columns_2]

independent_variables = sm.add_constant(independent_variables)

regression_model = sm.OLS(dependent_variables,independent_variables).fit()
regression_model.summary()

x_labels = [
    0.05,
    0.06,
    0.18,
    0.23,
    1.67,
]

regression_model.params

coeff = regression_model.conf_int()

coeff_sorted =coeff.sort_values(1, ascending=True)

coeff_sorted[1].plot(kind='bar')

plt.xlabel('Variables')
plt.ylabel('Coefficient')

plt.title('Variable Coefficients (Stay Length as constant)')
```




    Text(0.5, 1.0, 'Variable Coefficients (Stay Length as constant)')




    
![png](output_95_1.png)
    



```python
### INSIGHTS
⮞ SOI (Severity of Illness) correlates the most with stay with a coefficient of 0.23, followed by TOA (Type of Admission) (0.18), Department (0.06), and Age (0.05).
```


      Input In [518]
        ⮞ SOI (Severity of Illness) correlates the most with stay with a coefficient of 0.23, followed by TOA (Type of Admission) (0.18), Department (0.06), and Age (0.05).
        ^
    SyntaxError: invalid character '⮞' (U+2B9E)
    



```python

```


```python
## Key Insights and Recommendations:

Insight: Patients who stay for the two most common Stay Lengths of 11-20 and 21-30 Days also have the highest average Admission Deposits
Recommendation: Decrease patient stays of 31+ Days to increase profits

Insight: Regression Analysis revealed that Severity of Illness is the best predictor of how long a patient will need to stay,  but is still considered a low correlation.
Patients identified as Severity of Illness  of Extremely ill at intake were more likely to have a longer stay, and patients identified as Minorly ill at intake were more likely to have a shorter stay.
Recommendation: Although a low correlation, this information can still be used to predict stays for those who are Extremely and Minorly ill. See "Analyst's Next Steps" for further clarification.

Insight:  Type of Admission has the second strongest correlation of 0.18, which is still considered a low correlation.
Patients with the Type of Admission of Urgent were more likely to have stays of 0-20 Days, and patients with Emergency admissions were not likely to have stays 61+ Days. 
Recommendation:  Although a low correlation, this information can still be used to predict stays for those who were Urgent and Emergency Admissions. See "Analyst's Next Steps" for further clarification.

```


      Input In [519]
        Insight: Patients who stay for the two most common Stay Lengths of 11-20 and 21-30 Days also have the highest average Admission Deposits
                          ^
    SyntaxError: invalid syntax
    



```python

```


```python
## Key Insights and Next Steps:

Insight: Variables Age Groups and Departments have a very low correlation with Stay Length, although some groups and departments seem to correlate more than others
Analyst's Next Step: The Age Groups and Departments that have low variability could be further analyzed to see if correlations can be identified to refine predictions.

Insight: The two highest correlated variables of Severity of Illness and Type of Admission are still considered low correlation.
Analyst's Next Step: Analyze these variables in conjunction with each other and Stay Length to determine if insights can be gleaned from these three variables together, particularly the less variable categories within the variables.

Insight: An abundance of categorical data and lack of numerical data made statistical analysis difficult, and contributed to the lack of success with this project. 
Analyst's Next Step: More numeric OR categorical (nominal or ordinal) data that can be binary or trinary should be sought out to conduct a more thorough and predictive analysis.
```


      Input In [520]
        Insight: Variables Age Groups and Departments have a very low correlation with Stay Length, although some groups and departments seem to correlate more than others
                           ^
    SyntaxError: invalid syntax
    

