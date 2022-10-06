#!/usr/bin/env python
# coding: utf-8

# In[447]:


# Predicting Hospital Patient Length of Stay


# In[ ]:





# In[448]:


### Since the beginning of the COVID-19 pandemic, hospitals have consistently struggled with patient bed, resource, and staffing shortages, and being able to forecast patient stay length will help hospitals to predict bed/room availability, age-appropriate care, staffing, and resource needs, as well as ensure financial security.


# In[ ]:





# In[449]:


## Issues


# In[450]:


### Issue: Ongoing financial instability is a problem across America's hospital as they recover from the height of the COVID-19 pandemic.


# In[451]:


### Solution: Hospitals will be better able to predict revenue to ensure financial security through patient Length of Stay prediction.


# In[ ]:





# In[452]:


### Issue: Age is a determinant of educational and counseling strategies as well as recovery outcome, and hospitals cannot prepare for this without a forecast of age care needs. 


# In[453]:


### Solution: Hospitals will be better able to forecast needs for age appropriate care through patient Length of Stay prediction.


# In[ ]:





# In[454]:


### Issue: The nurse shortage in America is only increasing, and miscommunication in staffing needs is a contributor.


# In[455]:


### Solution: Hospitals will be better able to forecast staffing needs through patient Length of Stay prediction.


# In[ ]:





# In[456]:


### Issue: The COVID-19 pandemic shined a light on how the healthcare system struggles when faced with a lack of patient beds and resources. 


# In[457]:


### Solution: Hospitals will be better able to forecast bed/resource needs through patient Length of Stay prediction.


# In[ ]:





# In[458]:


## Importing extensions


# In[459]:


import numpy as np


# In[460]:


import pandas as pd


# In[461]:


import matplotlib.pyplot as plt


# In[462]:


import seaborn as sns


# In[463]:


import statsmodels.api as sm
sns.set_style("darkgrid")


# In[464]:


## Uploading tables 


# In[465]:


# <span style="color:blue">Descriptive Statistics</span>


# In[466]:


## Train table


# In[467]:


train = pd.read_csv("train_data.csv")
#train.info()
#train.describe()


# In[468]:


train.head()


# In[469]:


#print(train.isnull().sum())


# In[470]:


### Dropping columns with null values 


# In[471]:


print(train.shape)


# In[472]:


train.drop("Bed Grade", axis= "columns", inplace=True)


# In[473]:


print(train.shape)


# In[474]:


train.drop("City_Code_Patient", axis= "columns", inplace=True)


# In[475]:


print(train.shape)


# In[476]:


### Identifying data types of each column to check compatibility


# In[477]:


train.dtypes


# In[ ]:





# In[478]:


## Looking at Stay Length


# In[479]:


### Counting the unique values in 'Stay'

Stay_count = train['Stay'].value_counts(sort= True)
Stay_count


# In[480]:


### Expressing the counts as proportions
print(train.Stay.value_counts(normalize= True))


# In[481]:


### Visualizing counts
sns.set_style("darkgrid")

category_order = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', 'More than 100 Days']

sns.countplot(y= "Stay", data= train, order= category_order)

plt.title('Stay Length of Patients (Days)')


# In[482]:


## INSIGHTS:
⮞ Stay Lengths of 11-20 and 21-30 Days are most common, while Stay Lengths suddenly increase at 51-60 Days
⮞ 11-20 Days: 78,139 patients, an increase of 🡹 54,525 compared to 0-10 Days
⮞ 21-30 Days: 87,491 patients, an increase of 🡹 9,352 compared to 11-20 Days
⮞ 51-60 Days: 35,018 patients, an increase of 🡹 23,275 compared to 41-50 Days, and 🡹 32,274 compared to 61-70 Days


# In[ ]:





# In[483]:


## Looking at Ages by Stay Length


# In[484]:


### Counting the unique values in 'Age'

Stay_count = train['Stay'].value_counts(sort= True)
Stay_count


# In[485]:


### Expressing the counts as proportions
print(train.Stay.value_counts(normalize= True))


# In[486]:


### Age distribution for each stay length
stay_age_distribution = pd.get_dummies(train.loc[:,['Stay','Age']], columns=['Age']).groupby('Stay', as_index=False).sum()
stay_age_distribution


# In[487]:


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


# In[488]:


## INSIGHTS
⮞ Majority of Admissions: Age Groups 31-40 (40,859), 41-50 (40,054), 51-60 (30,143)
⮞ 11-20 Days: Age Groups 11-20 🡹, 51-60 🡹, and 61-70 🡹 from previous groups
⮞ 21-30 Days: Age Groups 0-10 🡻, 11-20 🡻, and 81-90 🡹 from previous groups
⮞ 51-60 Days: Age Groups 41-50 🡹, 51-60 🡹, and 61-70 🡹 from previous groups
⮞ All Age Groups increase at Stay Length of 51-60 Days, which reflects the increase when looking at overall Stay Lengths, with Age Group 81-90 the most responsible 


# In[ ]:





# In[ ]:


## Looking at Departments by Stay Length


# In[489]:


### Counting the unique values in 'Department'
Dept_count = train['Department'].value_counts(sort= True)
Dept_count


# In[490]:


### Expressing the counts as proportions
Dept_count = train['Department'].value_counts(normalize= True)
Dept_count


# In[491]:


### Department distribution for each stay length
stay_dept_distribution = pd.get_dummies(train.loc[:,['Stay','Department']], columns=['Department']).groupby('Stay', as_index=False).sum()
stay_dept_distribution


# In[492]:


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


# In[493]:


## INSIGHTS:
⮞ Gynecology has the most admitted patients (249,486), followed by anesthesia (29,649), radiotherapy (28,516),
TB & Chest disease (9,586), surgery (1,201)
⮞ The most common Stay Length among all Departments is 21-30 Days, followed by 11-20 Days
⮞ Surgery has the largest ratio of patients who stay more than 100 Days


# In[ ]:





# In[ ]:


## Severity of Illness by Stay Length


# In[494]:


### Counting unique values in "Severity of Illness"
SIL_count = train['Severity of Illness'].value_counts(sort= True)
SIL_count


# In[495]:


### Expressing the counts as proportions
print(train['Severity of Illness'].value_counts(normalize= True))


# In[496]:


### Severity of Illness distribution for each stay length
stay_SIL_distribution = pd.get_dummies(train.loc[:,['Stay','Severity of Illness']], columns= ['Severity of Illness']).groupby('Stay', as_index=False).sum()
stay_SIL_distribution


# In[497]:


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


# In[498]:


### INSIGHTS:
⮞The majority of patients are identified as Moderately ill (175,843 patients), followed by Minorly ill (85,872 patients), and finally Extreme illness (56,723 patients)
⮞ There does not seem to be a pattern to Stay Length in the patients identified as Moderately ill
⮞ Patients identified as Minorly ill seem to stay for a shorter amount of time, with a larger amount staying 0-10 Days, and smaller amount staying More than 100 Days
⮞  Patients identified as Extremely ill seem to stay for a longer amount of time, with a smaller amount staying 0-10 Days, and a larger amount staying More than 100 Days


# In[ ]:





# In[499]:


## Type of Admission by Stay Length


# In[500]:


### Counting unique values in "Type of Admission"
TOA_count = train['Type of Admission'].value_counts(sort= True)
TOA_count


# In[501]:


### Expressing the counts as proportions
print(train['Type of Admission'].value_counts(normalize= True))


# In[502]:


# TOA distribution for each stay length
stay_TOA_distribution = pd.get_dummies(train.loc[:,['Stay','Type of Admission']], columns=['Type of Admission']).groupby('Stay', as_index=False).sum()
stay_TOA_distribution


# In[503]:


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


# In[504]:


### INSIGHTS:
⮞The leading Type of Admission is Trauma (152261,~47.81%), followed by Emergency (117676, ~36.95%), and lastly Urgent (48501, ~15.23%)
⮞ 58958, or ~66% of Emergency, 79989, or ~71% of Trauma, and 26683, or ~72% of Urgent Admissions stay between 11-20 to 21-30 days
⮞ Emergency Admissions are more likely to stay between 0-10 days to 11-20 days than any other group


# In[ ]:





# In[505]:


## Admission Deposit by Stay Length


# In[506]:


### Identifying the average overall Admission Deposit
AD_Avg = train['Admission_Deposit'].mean()
AD_Avg


# In[507]:


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


# In[508]:


### INSIGHTS

⮞ There is not much variance in the Admission Deposits (AD) across all Stay Lengths
⮞ The average overall AD was $4880.75
⮞ Stay Lengths of 21-30 days had the highest average AD of ~$5000
⮞ Stay Lengths of 81-90 days had the lowest average AD of ~$4500
⮞ Stay Lengths of 61-70 days had the largest range of data
⮞ After 0-10 days ADs increase overall, and after 41-50 days ADs decrease overall


# In[ ]:





# In[ ]:


# <span style="color:blue">Inferential Statistics</span>


# In[ ]:


## Regression Analysis


# In[ ]:


### Importing Train data that has been converted into Nominal and Ordinal Variables for statistical analysis
stats = pd.read_csv("train_data2.csv")


# In[ ]:


stats.head()


# In[ ]:


### Dropping irrelevant columns


# In[ ]:


stats.drop("Regression formula", axis= "columns", inplace=True)


# In[ ]:


stats.drop("Unnamed: 6", axis= "columns", inplace=True)


# In[ ]:


stats.drop("Unnamed: 7", axis= "columns", inplace=True)


# In[ ]:


stats.head()


# In[509]:


### Correlation Heatmap to identify variable correlation

columns = ['TOA', 'Department', 'SOI','Age', 'Stay']
corr_df = stats[columns].corr()
df_heatmap = sns.heatmap(corr_df, cmap='YlOrRd', annot=True)
plt.title('Correlation Heatmap')


# In[510]:


## INSIGHT
⮞ SOI (Severity of Illness) (.09) and TOA (Type of Admission) (0.08) seem to have a slight correlation with Stay Length.


# In[ ]:





# In[511]:


### Visualizing Variable Correlation with Coefficient plot

col_list_corr = corr_df.columns
corr_df=corr_df.sort_values('Stay', ascending=False)
corr_df['Stay'].plot(kind='bar')
plt.xlabel('Variables')
plt.ylabel('Coefficient')
plt.title('Variable Correlation with Stay Length')


# In[512]:


## INSIGHT
⮞ This confirms that SOI (Severity of Illness) (.09) and TOA (Type of Admission) (.08) seem to have a slight correlation with Stay Length.


# In[ ]:





# In[513]:


### OLS Regression Model

columns_1 = ['TOA', 'Department', 'SOI','Age']
columns_2 = ['Stay']

independent_variables = stats[columns_1]
dependent_variables = stats[columns_2]

independent_variables = sm.add_constant(independent_variables)

regression_model = sm.OLS(dependent_variables,independent_variables).fit()
regression_model.summary()


# In[514]:


### INSIGHTS
⮞  R Squared (~.02%), suggesting this model is a poor fit. 


# In[ ]:





# In[517]:


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


# In[518]:


### INSIGHTS
⮞ SOI (Severity of Illness) correlates the most with stay with a coefficient of 0.23, followed by TOA (Type of Admission) (0.18), Department (0.06), and Age (0.05).


# In[ ]:





# In[519]:


## Key Insights and Recommendations:

Insight: Patients who stay for the two most common Stay Lengths of 11-20 and 21-30 Days also have the highest average Admission Deposits
Recommendation: Decrease patient stays of 31+ Days to increase profits

Insight: Regression Analysis revealed that Severity of Illness is the best predictor of how long a patient will need to stay,  but is still considered a low correlation.
Patients identified as Severity of Illness  of Extremely ill at intake were more likely to have a longer stay, and patients identified as Minorly ill at intake were more likely to have a shorter stay.
Recommendation: Although a low correlation, this information can still be used to predict stays for those who are Extremely and Minorly ill. See "Analyst's Next Steps" for further clarification.

Insight:  Type of Admission has the second strongest correlation of 0.18, which is still considered a low correlation.
Patients with the Type of Admission of Urgent were more likely to have stays of 0-20 Days, and patients with Emergency admissions were not likely to have stays 61+ Days. 
Recommendation:  Although a low correlation, this information can still be used to predict stays for those who were Urgent and Emergency Admissions. See "Analyst's Next Steps" for further clarification.


# In[ ]:





# In[520]:


## Key Insights and Next Steps:

Insight: Variables Age Groups and Departments have a very low correlation with Stay Length, although some groups and departments seem to correlate more than others
Analyst's Next Step: The Age Groups and Departments that have low variability could be further analyzed to see if correlations can be identified to refine predictions.

Insight: The two highest correlated variables of Severity of Illness and Type of Admission are still considered low correlation.
Analyst's Next Step: Analyze these variables in conjunction with each other and Stay Length to determine if insights can be gleaned from these three variables together, particularly the less variable categories within the variables.

Insight: An abundance of categorical data and lack of numerical data made statistical analysis difficult, and contributed to the lack of success with this project. 
Analyst's Next Step: More numeric OR categorical (nominal or ordinal) data that can be binary or trinary should be sought out to conduct a more thorough and predictive analysis.

