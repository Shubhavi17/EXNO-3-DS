## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
  ```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
<img width="406" height="416" alt="image" src="https://github.com/user-attachments/assets/2446f85c-e9f1-4ce8-95bc-894c81b47abf" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="331" height="241" alt="image" src="https://github.com/user-attachments/assets/d6337fbb-c262-4df3-99b2-07cdfa11084a" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="490" height="428" alt="image" src="https://github.com/user-attachments/assets/b16dd1c5-52b4-4cdb-81a4-37d3052deda3" />

```
 le=LabelEncoder()
 dfc=df.copy()
 dfc['ord_2']=le.fit_transform(dfc['ord_2'])
 dfc
```
<img width="539" height="438" alt="image" src="https://github.com/user-attachments/assets/d62f3b54-8fa6-44d9-83ef-3fca880d69e1" />

```
 from sklearn.preprocessing import OneHotEncoder
 ohe=OneHotEncoder()
 df2=df.copy()
 enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
 df2=pd.concat([df2,enc],axis=1)
 df2
```
<img width="799" height="414" alt="image" src="https://github.com/user-attachments/assets/9d25fc9e-2916-4a3b-b833-4b9abd2348e7" />

```
 pd.get_dummies(df2,columns=["nom_0"])
```
<img width="833" height="498" alt="image" src="https://github.com/user-attachments/assets/200a76b3-b057-4fde-9371-33449fcb2891" />

```
 pip install--upgrade category_encoders
```
<img width="824" height="325" alt="image" src="https://github.com/user-attachments/assets/0e2b1754-95f1-4dab-90c5-6aa25246f164" />

```

 from category_encoders import BinaryEncoder
 df=pd.read_csv("data.csv")
 df
 be=BinaryEncoder()
 nd=be.fit_transform(df['Ord_2'])
 df
 dfb=pd.concat([df,nd],axis=1)
 dfb
```
<img width="787" height="393" alt="image" src="https://github.com/user-attachments/assets/a439c635-dd1f-4da0-9d7a-a00a8838d341" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="730" height="433" alt="image" src="https://github.com/user-attachments/assets/c4f5d74b-11d4-40bf-bd29-14047165661d" />

```
 import pandas as pd
 from scipy import stats
 import numpy as np
 df=pd.read_csv("Data_to_Transform.csv")
 df
```
<img width="821" height="403" alt="image" src="https://github.com/user-attachments/assets/214b8475-03c0-4645-8f3f-4d67ce83745d" />

```
df.skew()
```
<img width="608" height="342" alt="image" src="https://github.com/user-attachments/assets/8bc2556d-9b07-40b2-8f48-e245713f1220" />

```
 np.log(df["Highly Positive Skew"])
```
<img width="475" height="746" alt="image" src="https://github.com/user-attachments/assets/a4d8b45e-ec12-4b2d-8141-7588db0b0113" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="534" height="757" alt="image" src="https://github.com/user-attachments/assets/cfc33537-42c3-42bb-9c7c-920b4b6e4651" />

```
 np.sqrt(df["Highly Positive Skew"])
```
<img width="538" height="749" alt="image" src="https://github.com/user-attachments/assets/3b8a4732-234f-40a8-b2e7-696ab71aa2aa" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
 df
```
<img width="810" height="400" alt="image" src="https://github.com/user-attachments/assets/27f4136e-ace1-43c5-963e-ee5e6146eea3" />

```
 df.skew()
```
<img width="454" height="272" alt="image" src="https://github.com/user-attachments/assets/c6748993-2cbf-4159-a2e5-47814585fa92" />

```
 df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
 df.skew()
```
<img width="617" height="309" alt="image" src="https://github.com/user-attachments/assets/2e000a8e-8a7f-46c6-95ee-378b84bd0064" />

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 df
```
<img width="836" height="579" alt="image" src="https://github.com/user-attachments/assets/af6ddf5d-24de-4d62-b77e-a9d65771da70" />

```
 import seaborn as sns
 import statsmodels.api as sm
 import matplotlib.pyplot as plt
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```
<img width="775" height="582" alt="image" src="https://github.com/user-attachments/assets/2fdbab49-2a88-45a3-ba83-a093f0db13f6" />

```
 sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
 plt.show()
```
<img width="771" height="581" alt="image" src="https://github.com/user-attachments/assets/50f157fb-b586-4977-8952-62a5c927c009" />

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```
<img width="776" height="558" alt="image" src="https://github.com/user-attachments/assets/3b0b3a9f-0e48-4456-a71f-19781ede905e" />

```
 df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
 sm.qqplot(df["Highly Negative Skew"],line='45')
 plt.show()
```
<img width="799" height="601" alt="image" src="https://github.com/user-attachments/assets/4dad7620-a9f1-4925-a1c2-be6942139474" />

```
dt=pd.read_csv("titanic_dataset.csv")
 dt
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 dt["Age_1"]=qt.fit_transform(dt[["Age"]])
 sm.qqplot(dt['Age'],line='45') 
plt.show()
```
<img width="729" height="529" alt="image" src="https://github.com/user-attachments/assets/5a8773fe-7a3e-4f94-a0af-14449b487f6b" />

```
 sm.qqplot(df["Highly Negative Skew_1"],line='45')
 plt.show()
```
<img width="736" height="513" alt="image" src="https://github.com/user-attachments/assets/2277e70f-0598-4aa6-95cb-a59d292672e7" />



# RESULT:
       # INCLUDE YOUR RESULT HERE

       
