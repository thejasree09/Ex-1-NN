<H3>ENTER YOUR NAME: THEJA SREE G</H3>
<H3>ENTER YOUR REGISTER NO: 212224110056</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 27/01/2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
#importing libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Reading the dataset
df=pd.read_csv("Churn_Modelling.csv", index_col="RowNumber")
df

#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df

#Checking for null values
df.isnull().sum()

#Checking for duplicate values
df.duplicated()

#Describing the dataset
df.describe()

#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y

#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```


## OUTPUT:
## DataSet:
![Screenshot 2024-08-28 183829](https://github.com/user-attachments/assets/4bc18611-b41d-4a9e-bab0-deb7d86b745f)

## Dropping The Unwanted DataSet:
![Screenshot 2024-08-28 183842](https://github.com/user-attachments/assets/4a421686-308b-4393-aac9-a5f0d6bc27b7)

## Checking Null Values:
![Screenshot 2024-08-28 183856](https://github.com/user-attachments/assets/60c28982-c35b-4263-9344-d5e05d7b38de)

## Checking For Duplication:
![Screenshot 2024-08-28 183907](https://github.com/user-attachments/assets/529d8944-4041-4bf3-bc61-558458f057d3)

## Describing The DataSet:
![Screenshot 2024-08-28 183924](https://github.com/user-attachments/assets/1a512e87-5539-4a10-9413-2e33880e698f)

## Scaling The DataSet:
![Screenshot 2024-08-28 183940](https://github.com/user-attachments/assets/4aa3054c-8f0f-4586-b2e0-029a146081b0)

## X Features:
![Screenshot 2024-08-28 183954](https://github.com/user-attachments/assets/82d65f63-a88d-4608-9ce5-a0f5ea4a8b52)

## Y Features:
![Screenshot 2024-08-28 184007](https://github.com/user-attachments/assets/0e00c065-b14d-46b3-bee6-24029c013851)

## Splitting The Training And Testing DataSet:
![Screenshot 2024-08-28 184020](https://github.com/user-attachments/assets/191f8377-5133-430a-be60-fe5cc79cf611)



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


