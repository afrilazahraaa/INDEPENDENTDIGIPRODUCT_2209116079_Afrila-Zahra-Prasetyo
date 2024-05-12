import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('Data_Cleaning.csv')  # Ganti 'Data_Cleaning.csv' dengan nama file dataset Anda

# Visualisasi EDA
st.title('Exploratory Data Analysis (EDA)')

# Menampilkan data preview
st.subheader('Data Preview')
st.dataframe(df.head())

# Histogram untuk Unit price
st.subheader('Unit Price Distribution')
fig, ax = plt.subplots()
sns.histplot(df['Unit price'], kde=True, ax=ax)
st.pyplot(fig)

# Bar plot untuk Quantity
st.subheader('Quantity per Product Line')
fig, ax = plt.subplots()
sns.barplot(x='Quantity', y='Product line', data=df, ax=ax)
st.pyplot(fig)

# Pie chart untuk Gender
st.subheader('Gender Distribution')
gender_count = df['Gender'].value_counts()
fig, ax = plt.subplots()
ax.pie(gender_count, labels=gender_count.index, autopct='%1.1f%%', startangle=140)
ax.axis('equal')
st.pyplot(fig)

# Bar plot untuk Payment
st.subheader('Payment Method Distribution')
payment_count = df['Payment'].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=payment_count.index, y=payment_count.values, ax=ax)
ax.set_xlabel('Payment Method')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# Perform clustering
st.subheader("Performing Clustering")

# Select numerical features for clustering
numerical_features = ['Unit price', 'Quantity']

# Prepare data for clustering
clustering_data = df[numerical_features].copy()

# Scaling the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)

# Perform KMeans clustering
num_clusters = 3  # Number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)



# Explanation of clusters
st.subheader("Explanation of Clusters")
st.write("Cluster 0: Explanation of Cluster 0")
st.write("Cluster 1: Explanation of Cluster 1")
st.write("Cluster 2: Explanation of Cluster 2")

# Count payment methods by gender in each cluster
payment_gender_counts = df.groupby(['Cluster', 'Gender', 'Payment']).size().unstack(fill_value=0)

# Display payment method distribution by gender in each cluster
st.subheader("Payment Method Distribution by Gender in Each Cluster")
st.dataframe(payment_gender_counts)

# Plot payment method distribution by gender in each cluster
st.subheader("Payment Method Distribution by Gender in Each Cluster (Visualization)")

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot payment method distribution by gender in each cluster
payment_gender_counts.plot(kind='bar', ax=ax)

# Set plot labels and title
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Payment Method Distribution by Gender in Each Cluster')

# Rotate x-axis labels
plt.xticks(rotation=0)

# Show plot
st.pyplot(fig)

# Count product line by unit price in each cluster
product_line_price_counts = df.groupby(['Cluster', 'Product line', 'Unit price']).size().unstack(fill_value=0)

# Display product line distribution by unit price in each cluster
st.subheader("Product Line Distribution by Unit Price in Each Cluster")
st.dataframe(product_line_price_counts)

# Plot product line distribution by unit price in each cluster
st.subheader("Product Line Distribution by Unit Price in Each Cluster (Visualization)")

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot product line distribution by unit price in each cluster
product_line_price_counts.plot(kind='bar', ax=ax)

# Set plot labels and title
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Product Line Distribution by Unit Price in Each Cluster')

# Rotate x-axis labels
plt.xticks(rotation=0)

# Show plot
st.pyplot(fig)
