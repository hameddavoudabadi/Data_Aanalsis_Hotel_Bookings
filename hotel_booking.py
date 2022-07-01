import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data_filepath = "hotel_bookings.csv"
df = pd.read_csv(data_filepath)
print("\ndf.columns \n=",df.columns)
print("\ndf.len =\n",len(df))
print("\ndf.shape =\n",df.shape)
print("\ndf.size =\n",df.size)
print("\ndf.dtypes =\n",df.dtypes)
print("\ndf.describe\n",df.describe())

# checking null values
print("\ndf.isnull =\n", df.isnull().sum())
sns.heatmap(df.isnull())
# it seems that we have too much nulls in "agent" and "company" columns and we have to exclude these two columns from our data.

# removing null values from country and children columns.
df = df.dropna(subset = ['country'])
df = df.dropna(subset = ['children'])
df.isnull().sum()

# To find the unique values in each column
for (columnName, columnData) in df.iteritems():
    print('Column Name : ', columnName)
    print(df[columnName].unique())

# we have three columns with "Undefind" values: meal - market_segment - distribution_channel

# Removing Undefined values
df2 = df[df["distribution_channel"].str.contains("Undefined")==False]
df2 = df[df["market_segment"].str.contains("Undefined")==False]
df2 = df[df["meal"].str.contains("Undefined")==False]

print("\ndf.shape =\n",df2.shape)

# adding children to adults to creat total guests
df2['total_guests'] = df2['adults'] + df2['children']

# it does not make sense that a booking has 0 guest. So i will drop the rows with 0 guest.
df2 = df2[df2['total_guests'] != 0]

# adding a total stays nights by adding stays_in_weekend_nights to stays_in_week_nights
df2['total_stays'] = df2['stays_in_weekend_nights'] + df2['stays_in_week_nights']

df2 = df2.reset_index(drop=True)
print("\ndf.shape =\n",df2.shape)

# replacing the month string values with numbers
df2['arrival_date_month'].replace({'January' : '1',
        'February' : '2',
        'March'    : '3',
        'April'    : '4',
        'May'      : '5',
        'June'     : '6',
        'July'     : '7',
        'August'   : '8',
        'September': '9',
        'October'  : '10',
        'November' : '11',
        'December' : '12'}, inplace=True)
df2['arrival_date_month'].unique()

# Showing types of Hotel has been booked
plt.figure(figsize=(5,5))
sns.countplot(x='hotel', data = df2)
plt.title('Types of Hotels', weight='bold')
plt.xlabel('Hotel', fontsize=10)
plt.ylabel('Count', fontsize=10)

# How many of bookings has been canceled?
plt.figure(figsize=(5,5))
sns.countplot(x='is_canceled', data = df2)
plt.title('Canceled bookings (0=Not, 1=Canceled)', weight='bold')
plt.xlabel('is_canceled', fontsize=10)
plt.ylabel('Count', fontsize=10)

# Showing the distribution of canceled bookins around lead time for different years.
# Lead time = Number of days that elapsed between the date of the booking and the arrival date
plt.figure(figsize=(10,10))
sns.violinplot(x='arrival_date_year', y ='lead_time', hue="is_canceled", data=df2, split = True)
sns.despine(left=True)
plt.title('Arrival Year vs Lead Time vs Canceled Situation', weight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Lead Time', fontsize=12)

# Crosstab for relationship between foods reserved and type of Hotels
pd.crosstab(df2['hotel'], df2['meal'], margins=True, margins_name = 'Total', normalize= True).iloc[:10].round(2)*100


# Crosstab for relationship between type of rooms which was reserved and type of rooms assigned
pd.crosstab(df2['reserved_room_type'], df2['assigned_room_type'],normalize='index').round(2)*100

plt.figure(figsize=(5,5))   #*
sns.countplot(data=df2, x = 'is_canceled', hue='is_repeated_guest')
plt.title('Do the repeated guests canceled the booking?')
# the repeated guests did not cancel the booking

#`arrival_date_month` exploration
df2['arrival_date_month']= df2['arrival_date_month'].astype('int64')
plt.figure(figsize=(8,5))
sns.countplot(x='arrival_date_month', data = df2, order=pd.value_counts(df2['arrival_date_month']).index)
plt.title('Arrival Month', weight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Count', fontsize=12)
# August is the most crowded month

plt.figure(figsize=(5,5))   #*
sns.lineplot(data=df2, x= 'arrival_date_month', y = 'adr', hue='hotel',)
plt.xticks(rotation=45,fontsize=10)
# the adr, which is related to the price of the room, increases the most in the crowdest month

group_data = df2.groupby([ 'arrival_date_month','is_canceled']).size().unstack(fill_value=0)
group_data.sort_values('arrival_date_month', ascending = True).plot(kind='bar',stacked=True, figsize=(7,5))
plt.title('Arrival Month vs is_canceled', weight='bold')
plt.xlabel('Arrival Month', fontsize=12)
plt.xticks(rotation=360)
plt.ylabel('Count', fontsize=12)

# Create Top 10 Countries with the most bookings
plt.figure(figsize=(5,5))
sns.countplot(x='country', data=df2, order=pd.value_counts(df2['country']).iloc[:10].index)
plt.title('Top 10 Countries with the most bookings', weight='bold')
plt.xlabel('Country', fontsize=10)
plt.ylabel('Count', fontsize=10)

# Using Label Encoder method for categorical features
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df2['hotel'] = labelencoder.fit_transform(df2['hotel'])
df2['arrival_date_month'] = labelencoder.fit_transform(df2['arrival_date_month'])
df2['meal'] = labelencoder.fit_transform(df2['meal'])
df2['country'] = labelencoder.fit_transform(df2['country'])
df2['market_segment']= labelencoder.fit_transform(df2['market_segment'])
df2['distribution_channel']=labelencoder.fit_transform(df2['distribution_channel'])
df2['is_repeated_guest'] = labelencoder.fit_transform(df2['is_repeated_guest'])
df2['reserved_room_type'] = labelencoder.fit_transform(df2['reserved_room_type'])
df2['assigned_room_type'] = labelencoder.fit_transform(df2['assigned_room_type'])
df2['deposit_type'] = labelencoder.fit_transform(df2['deposit_type'])
df2['agent'] = labelencoder.fit_transform(df2['agent'])
df2['customer_type'] = labelencoder.fit_transform(df2['customer_type'])
df2['reservation_status'] = labelencoder.fit_transform(df2['reservation_status'])


# Creating a new dataframe for categorical data
df_cat = df2[['hotel','is_canceled','arrival_date_month','meal',
                                     'country','market_segment','distribution_channel',
                                     'is_repeated_guest', 'reserved_room_type',
                                     'assigned_room_type','deposit_type','agent',
                                     'customer_type','reservation_status']]
df_cat.info()

# Creating a new dataframe for numerical data
df_num= df2.drop(['hotel','is_canceled', 'arrival_date_month','meal',
                                       'country','market_segment','distribution_channel',
                                       'is_repeated_guest', 'reserved_room_type',
                                       'assigned_room_type','deposit_type','agent',
                                       'customer_type','reservation_status'], axis = 1)
df_num.info()

# Checking the correlation between Categorical Data by heatmap
cor_matrix_cat = df_cat.corr()
plt.figure(figsize=(10,10))
cat_cor_mask = np.triu(np.ones_like(cor_matrix_cat))
sns.heatmap(cor_matrix_cat, annot=True, fmt=".2f", cmap='RdBu', mask= cat_cor_mask, vmin=-1, vmax=1, center= 0, square=True, linewidths=2, cbar_kws={"shrink": .5}).set(ylim=(17, 0))
plt.title("Correlation Matrix for Categorical Data ",size=15, weight='bold')
# the reservation_status has a high reverse correlation with is_canceled and it seemed we have to delete that column

# Checking the correlation between Numerical heatmap
plt.figure(figsize=(12,12))
cor_matrix_num = df_num.corr()
mask_numerical = np.triu(np.ones_like(cor_matrix_num, dtype=np.bool))
sns.heatmap(cor_matrix_num, annot=True, fmt=".2f", cmap='RdBu', mask= mask_numerical, vmin=-1, vmax=1, center= 0, square=True, linewidths=2, cbar_kws={"shrink": .5}).set(ylim=(17, 0))
plt.title("Correlation Matrix for Numerical Data ",size=15, weight='bold')
# babies has not an important correlation.

df3 = df2.drop(['reservation_status', 'babies', 'reservation_status_date','company'], axis=1)

# Building the sample group
y_model = df2.iloc[:,1]
X_model = pd.concat([df2.iloc[:,0],df3.iloc[:,2:]], axis=1)
y_model.describe()

# Split to train and test with 70-30 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.3, random_state=42)

# Implement standart scaler method
from sklearn.preprocessing import StandardScaler
standardScalerX = StandardScaler()
X_train = standardScalerX.fit_transform(X_train)
X_test = standardScalerX.fit_transform(X_test)

# Decision Tree Model Building
from sklearn.tree import DecisionTreeClassifier
decision_tree_m = DecisionTreeClassifier(criterion= 'gini', min_samples_split=8, min_samples_leaf = 4, max_features = 'auto')
# fitting the model
decision_tree_m.fit(X_train, y_train)

from sklearn.metrics import classification_report
# Predicting Model
predict_decision_tree = decision_tree_m.predict(X_test)

# Classification Reports
print("Decision Tree Classification: \n",classification_report(y_test, predict_decision_tree))
print("Accuracy on test data: {:.2f}".format(decision_tree_m.score(X_test, y_test)))
plt.show()
print("DONE")