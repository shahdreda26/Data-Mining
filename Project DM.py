#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd # for dataframes
import matplotlib.pyplot as plt 
from collections import Counter
import numpy as np
import seaborn as sns # for plotting graphs
sns.set_style('dark')
get_ipython().system('pip install scikit-learn-extra')
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import cophenet
from sklearn.metrics import silhouette_score


# In[15]:


data_set = pd.read_csv("E-commerce .csv")  #read of data set
data_set  #display


# In[16]:


data_set.head(10) # display first 10 raw


# In[17]:


data_set.tail(10) ##display  last 10 raw


# In[18]:


data_set.sample(10) #display 10 random raw


# In[19]:


#more information about each col. in data
data_set.info()


# In[20]:


data_set.describe(include="all") # describe all data


# In[21]:


data_set.describe(include="number") # describe all numeric data


# In[22]:


data_set.describe(include="object")  # describe all non numeric data


# In[23]:


#preprocessing
#outlieurs
columns = [ 'Total Spend' , 'Items Purchased' , 'Average Rating' , 'Days Since Last Purchase' , 'Age' ]#list include cols. because  check on outlieurs

# Identify and remove outliers
for column in columns:
    
    plt.figure(figsize=(5, 3)) #determine dimensions graph
    data_set.boxplot(column=column) 
    plt.title(column)  # Set the title of the boxplot
    plt.ylabel(column)  # Set the label for the y-axis
    plt.show()  # Display the boxplot


# In[24]:


#preprocessing
data_set.isnull() #show each col. contain null (true or false)


# In[25]:


data_set.isnull().sum() # display null in each col.


# In[26]:


data_set.isnull().sum().sum() #total number of missing value


# In[27]:


data_set.dropna(inplace=True) #remove null
data_set


# In[28]:


data_set.isnull().sum().sum() #total number of missing value


# In[29]:


data_set.isnull().sum() # display null in each col.


# In[30]:


data_set.duplicated() #show each col. contain duplicate (true or false) 


# In[31]:


data_set.duplicated().sum() #total number of duplicating


# In[32]:


#check on type (col).
print(data_set['Age'].dtype)


# In[33]:


print(data_set['Customer ID'].dtype)


# In[34]:


print(data_set['Days Since Last Purchase'].dtype)


# In[35]:


print(data_set['Items Purchased'].dtype)


# In[36]:


print(data_set['Items Purchased'].dtype)


# In[37]:


#outlieurs
for x in data_set.index: #عشان يمشى جوه العمود ويعدى على كل الفيم
    if data_set.loc[x,"Gender"]!= "Male" : #هنا عايزه يشوف لو نوعه غير الاتنين دول هيحذفه 
      if data_set.loc[x,"Gender"]!= "Female":
        data_set.drop(x,inplace = True) #remove 
data_set          


# In[38]:


#inconsestince
gender_mapping = { # هنا عشان يوحد طريقه الكتابه الى داخله 
    'Male': 'Male',
    'Female': 'Female',
    'M': 'Male',
    'F': 'Female',
    'Other': 'Other'
}

# Map inconsistent gender values to standardized values
data_set['Gender'] = data_set['Gender'].map(gender_mapping)
data_set


# In[39]:


#data inconsistence
data_set['Satisfaction Level'] = [x.title() if isinstance(x, str) else x for x in data_set['Satisfaction Level']]
data_set # هنا عشان اوحد طريقه الكتابه واخلى اول حرف من كل كلمه capital


# In[40]:


#inconsistence
data_set['City'] = [x.title() if isinstance(x, str) else x for x in data_set['City']]
data_set# هنا عشان اوحد طريقه الكتابه واخلى اول حرف من كل كلمه capital


# In[41]:


#data inconsistence
data_set['Membership Type'] = [x.title() if isinstance(x, str) else x for x in data_set['Membership Type']]
data_set# هنا عشان اوحد طريقه الكتابه واخلى اول حرف من كل كلمه capital


# In[42]:


#data inconsistence
data_set['Discount Applied'] = [x.title() if isinstance(x, str) else x for x in data_set['Discount Applied']]
data_set# هنا عشان اوحد طريقه الكتابه واخلى اول حرف من كل كلمه capital


# In[43]:


#choose two cols. #K_medoid( unsupervised learning) (numeric and nominal values) (Partitioning approach)
#  بستخدمه عشان يحط المتشابهه مع بعضه فى جروب واحد والمختلف فى جروب اخر 
Items_Purchased = data_set['Items Purchased']#   سواء الزياده بالموجب او السالب y يزداد x بعمل بينهم علاقه بحيث تزداد 
Total_Spend = data_set['Total Spend']
merged_col =pd.DataFrame({'x':Items_Purchased.values,'y':Total_Spend.values}) 
merged_col


# In[44]:


#preprocessing
merged_col.isnull() #show each col. contain null (true or false)


# In[45]:


merged_col.isnull().sum() # display null in each col.


# In[46]:


merged_col.isnull().sum().sum() #total number of missing value


# In[47]:


filtered_merged_col=merged_col.dropna(inplace=True) # remove null 
filtered_merged_col


# In[48]:


merged_col.duplicated()  #show each col. contain duplicate (true or false) 


# In[49]:


merged_col.duplicated().sum() #total number of duplicate value


# In[50]:


filtered_merged_col=merged_col.drop_duplicates( ) #remove duplicate


# In[51]:


filtered_merged_col #display data


# In[52]:


filtered_merged_col.shape #show number (raws-cols.)


# In[53]:


list=np.array(filtered_merged_col)
# Determine the optimal number of clusters

ssd = [] # Elbow Method
silhouette_scores = [] # silhouette Method
k_values = range(2, 11)  # Trying k values from 2 to 10
for k in k_values:
    kmedoids = KMedoids(n_clusters=k, random_state=42).fit(list) #do a fit for the data to apply the k_medoid
    ssd.append(kmedoids.inertia_)# add element in elbow
    silhouette_scores.append(silhouette_score(list, kmedoids.labels_)) #add element in silhouette 

# Find the elbow point
elbow_point = np.argmin(np.diff(ssd)) + 2  # Add 2 to offset the range starting from 2
optimal_k = k_values[np.argmax(silhouette_scores)]# Find the k value with the maximum silhouette score

# Apply the k-medoids clustering algorithm
kmedoids_silhouette = KMedoids(n_clusters=optimal_k, random_state=42)
kmedoids_silhouette.fit(list)
kmedoids_elbow = KMedoids(n_clusters=elbow_point, random_state=42)
kmedoids_elbow.fit(list)

# Get the cluster labels for each data point
cluster_labels_silhouette = kmedoids_silhouette.labels_
cluster_labels_elbow = kmedoids_elbow .labels_

# Print the results
print("Optimal number of clusters (Elbow Method):", elbow_point) #print number of optimal cluster (elbow point)
print('') #print empty line
print("Optimal number of clusters (Silhouette Score):", optimal_k)#print number of optimal cluster(Silhouette Score) 
print('')
print("Cluster Labels elbow:", cluster_labels_elbow)#print number cluster of each element (elbow point)
print('')
print("Cluster Labels optimal:", cluster_labels_silhouette) #print number cluster of each element(Silhouette Score)


# In[54]:


cluster_silhouette = kmedoids_silhouette.cluster_centers_ # determin centers of clusters (بيعرفى النقط الى هعمل عليها الحسابات فى المساله)
print("Clusters ",cluster_silhouette,'\n')
cluster_elbow = kmedoids_elbow.cluster_centers_
print("Clusters ",cluster_elbow,'\n')


# In[55]:


for j in range(k):
    for i in range(len(list)):
        if kmedoids_silhouette.labels_[i] == j:  # Check if the label of the current data point is equal to j
            x =list[i] # Access the row using iloc
            print('Clusters', j, ":" ,x) # printting of orderd pair is similarity in one cluster and dissimilarity in another clusters  


# In[56]:


for j in range(k):
    for i in range(len(list)):
        if kmedoids_elbow.labels_[i] == j:  # Check if the label of the current data point is equal to j
            x =list[i]  # Access the row using iloc
            print('Clusters', j, ":" , x)


# In[57]:


# Plot the clusters (assuming 'list' is a 2D array with each row representing a data point)
#Silhouette Score
plt.figure(figsize=(10, 7))
for i in range(kmedoids_silhouette.n_clusters):
    indices = np.where(cluster_labels_silhouette == i)
    plt.scatter(list[indices, 0], list[indices, 1], label='Cluster {}'.format(i+1))
plt.title('KMedoids Clusters (Optimal: {})'.format(optimal_k))
plt.legend()
plt.show()


# In[58]:


#elbow point
plt.figure(figsize=(10, 7))
for i in range(kmedoids_elbow.n_clusters):
    indices = np.where(cluster_labels_elbow == i)
    plt.scatter(list[indices, 0], list[indices, 1], label='Cluster {}'.format(i+1))
plt.title('KMedoids Clusters Elbow (Optimal: {})'.format(elbow_point))
plt.legend()
plt.show()


# In[59]:


#choose two cols. #Hierarchical clustering (Agglomerative (bottom-up)) Merge two closest clusters
Average_Rating = data_set['Average Rating']
Items_Purchased = data_set['Items Purchased']
merged_columns =pd.DataFrame({'X':Average_Rating.values,'Y':Items_Purchased.values})
merged_columns


# In[ ]:


#preprocessing
merged_columns.isnull() #show each col. contain null (true or false)


# In[ ]:


merged_columns.isnull().sum() # display null in each col.


# In[ ]:


merged_columns.isnull().sum().sum() #total number of missing value


# In[ ]:


filtered_merged_columns=merged_columns.dropna(inplace=True) #remove null
filtered_merged_columns


# In[ ]:


merged_columns.duplicated()  #show each col. contain duplicate (true or false)


# In[ ]:


merged_columns.duplicated().sum() #total number of duplicate


# In[ ]:


filtered_merged_columns=merged_columns.drop_duplicates( )# remove dublicate


# In[ ]:


filtered_merged_columns


# In[ ]:


filtered_merged_columns = np.array(filtered_merged_columns)
filtered_merged_columns


# In[ ]:


plt.figure(figsize=(10, 10))
plt.scatter(filtered_merged_columns[:, 0], filtered_merged_columns[:, 1], c='red')
for i in range(filtered_merged_columns.shape[0]):
    plt.annotate(str(i), xy=(filtered_merged_columns[i, 0], filtered_merged_columns[i, 1]), xytext=(3, 3), textcoords='offset points')
plt.xlabel("Average_Rating")
plt.ylabel("Items_Purchased")
plt.title("Scatter Plot of The Data")
plt.grid()
plt.show()


# In[ ]:


#accuracy
distances = pdist(filtered_merged_columns)#Compute the pairwise distances between observations

# Perform hierarchical clustering with different linkage methods
linkage_methods = ['single', 'complete', 'average', 'ward']
cophenetic_correlation = {}
for method in linkage_methods:
    Z = linkage(filtered_merged_columns, method=method, metric='euclidean')
    c, _ = cophenet(Z, distances)# calculate the cophenetic correlation  between the linkage matrix Z and the distance matrix distances
    #cophenet: function returns two values 
    # c : the cophenetic correlation coefficient 
    # _ : (blank variable) cophenetic distance matrix is discarded
    cophenetic_correlation[method] = c 

# Print the cophenetic correlation coefficient for each linkage method
for method, cophenetic_coef in cophenetic_correlation.items():
    print(f"Cophenetic Correlation Coefficient for {method} linkage method: {cophenetic_coef*100:.0f} %")


# In[ ]:


#caculate heuristic
single_linkage=linkage(filtered_merged_columns,method='single',metric='euclidean')# smallest distance 
complete_linkage=linkage(filtered_merged_columns,method='complete',metric='euclidean')# largest distance
average_linkage=linkage(filtered_merged_columns,method='average',metric='euclidean')# avg distance between 
ward_linkage=linkage(filtered_merged_columns,method='ward',metric='euclidean')


# In[ ]:


# draw dendrogram
plt.figure(figsize=(25,25))
plt.subplot(2,2,1),dendrogram(single_linkage),plt.title('Single')
plt.subplot(2,2,2),dendrogram(complete_linkage),plt.title('Complete')
plt.subplot(2,2,3),dendrogram(average_linkage),plt.title('Average')
plt.subplot(2,2,4),dendrogram(ward_linkage),plt.title('Ward')


# In[ ]:


fcluster1=fcluster(single_linkage,2)
print(f"Clusters{fcluster1}")


# In[ ]:


#pie chart (categorical value)
plt.figure(figsize=(12, 8)) # determine dimensions  graph
# merge graphs in one graph
plt.subplot(2, 2, 1) # determine location of graph

#data visualization
gender_counts = data_set['Gender'].value_counts()# Count the occurrences of each gender
colors = ['skyblue','pink']  # Define colors for each gender
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=colors,startangle=140,shadow=True) 
#autopct : بتخلينى احط النسب المئويه على الشكل
# colors :فيها الالوان الى حطاها listهتستدعى 
# shadow :عشان تعمل الشكل كانه بارز
plt.title('Gender Frequency')
plt.legend(title='Gender Frequency',loc='lower right')# عشان اعمل مربع يوضحلى الى جوه الرسم وبحدد عنوان ليه ومكانه 
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.subplot(2, 2, 2)
discount_applied_counts = data_set['Discount Applied'].value_counts()
colors = ['purple','lightgreen']  # Define colors for each gender
plt.pie(discount_applied_counts, labels=discount_applied_counts.index, autopct='%1.1f%%', colors=colors,startangle=140,shadow=True)
plt.title('Discount Applied Frequency')
plt.legend(title='Discount Applied',loc='lower right')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.subplot(2, 2, 3)
satisfaction_level_counts = data_set['Satisfaction Level'].value_counts()
colors = ['black','lightblue','yellow']  # Define colors for each gender
myexplode=[0,0,0.1] # low precentage # هنا بعمل جزء مقطوع من الجراف وبحدد نسبه بعده عنها
plt.pie(satisfaction_level_counts, labels=satisfaction_level_counts.index, autopct='%1.1f%%', colors=colors,startangle=140,
        explode=myexplode,shadow=True)
plt.title('Satisfaction Level Frequency')
plt.legend(loc='lower right',title='Satisfaction Level')

plt.subplot(2, 2, 4)
membership_type_counts = data_set['Membership Type'].value_counts()
colors = ['gold','silver','brown']  # Define colors for each gender
myexplode=[0,0,0.1]  # different precentage
plt.pie(membership_type_counts, labels=membership_type_counts.index, autopct='%1.1f%%', colors=colors,shadow=True,
        explode=myexplode)
plt.title('  Membership Type Frequency')
plt.legend(loc='lower left',title='Membership Type Frequency')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.tight_layout()
plt.show() # display graphs


# In[ ]:


#histogram (age) عشان يعرفنى كل عمر اتكرر كام مره 
plt.hist(data_set['Age'], bins=20, color='purple',alpha=0.4, edgecolor='white')
# edgecolor :عشان احدد كل عمود من بره
# alpha : بتحددلى درجه لون الجراف
# bins :  بحدد هقسم الداتا لاد ايه او لكام قيمه
plt.xlabel('Age') 
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.show()


# In[ ]:


#plot-->sum of (city Frequency)
plt.figure(figsize=(10, 6))
#similar of histogram
sns.countplot(x='City', data=data_set, palette='pastel')#هنا دا هيعد كل مدينه اتكررت كام مره 
plt.xlabel('City')
plt.ylabel('Count')
plt.title('City Frequency')
plt.show()


# In[ ]:


#scatter plot (show relationship between two cols)
plt.scatter(data_set['Items Purchased'], data_set['Total Spend'], color='maroon', alpha=0.4)
plt.title('Items Purchased & Total Spend')
plt.xlabel('Items Purchased')
plt.ylabel('Total Spend')
# Show the plot
plt.show()


# In[ ]:


plt.scatter(data_set['Average Rating'], data_set['Items Purchased'], alpha=0.5 , color="green")
plt.title(' Average Rating & Items Purchased')
plt.xlabel('Average Rating')
plt.ylabel('Items Purchased')
# Show the plot
plt.show()


# In[ ]:


grouped = data_set.groupby(['Membership Type', 'Satisfaction Level'])['Average Rating'].sum().unstack()
# Create the grouped column chart
plt.figure(figsize=(10, 7))
grouped.plot(kind='bar', rot=0)
plt.title('Grouped Column Chart Example')
plt.xlabel('Membership Type')
plt.ylabel('Average Rating')
plt.show()


# In[ ]:




