import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set the page configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Title and Description
st.title("🛒 Advanced Customer Segmentation App")
st.markdown("""
    This application allows you to perform **Customer Segmentation** using RFM analysis and clustering. 
    Upload your dataset, analyze the metrics, and visualize customer behaviors interactively.
""")

# Sidebar for uploading data
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1", dtype={'CustomerID': str, 'InvoiceID': str})
    st.sidebar.success("Dataset uploaded successfully!")
else:
    st.sidebar.warning("Please upload a CSV file to start!")
    st.stop()

# Data Cleaning and Preprocessing
st.header("🧹 Data Cleaning and Preprocessing")

# Create 'Amount' column
df["Amount"] = df["Quantity"] * df["UnitPrice"]
st.markdown("### Initial Data Preview")
st.write(df.head())

# Filter UK customers
df = df[df["Country"] == "United Kingdom"]
df = df[df["Quantity"] > 0]
df.dropna(subset=['CustomerID'], inplace=True)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["date"] = df["InvoiceDate"].dt.date

# Cleaned data preview
st.markdown("### Cleaned Data Overview")
st.write(df.describe())

# Summary Statistics
st.subheader("📊 Summary Statistics")
metrics = {
    "Number of Invoices": df['InvoiceNo'].nunique(),
    "Number of Products Bought": df['StockCode'].nunique(),
    "Number of Customers": df['CustomerID'].nunique(),
    "Average Quantity per Customer": round(df.groupby("CustomerID").Quantity.sum().mean(), 0),
    "Average Revenue per Customer (£)": round(df.groupby("CustomerID").Amount.sum().mean(), 2),
}
st.write(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

# Monthly Transactions Analysis
st.subheader("📅 Monthly Transactions Analysis")
df['month'] = df['InvoiceDate'].dt.month
monthly_counts = df.groupby('month').size()

# Plot using Plotly
fig_monthly = px.bar(
    monthly_counts,
    x=monthly_counts.index,
    y=monthly_counts.values,
    labels={"x": "Month", "y": "Transactions"},
    title="Transactions Per Month"
)
st.plotly_chart(fig_monthly)

# RFM Analysis
st.header("📈 RFM Analysis")

# Recency Calculation
now = pd.Timestamp("2011-12-09")
recency_df = df.groupby("CustomerID")["date"].max().reset_index()
recency_df["Recency"] = (now - pd.to_datetime(recency_df["date"])).dt.days

# Frequency Calculation
frequency_df = df.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
frequency_df.rename(columns={"InvoiceNo": "Frequency"}, inplace=True)

# Monetary Calculation
monetary_df = df.groupby("CustomerID")["Amount"].sum().reset_index()
monetary_df.rename(columns={"Amount": "Monetary"}, inplace=True)

# Combine RFM
rfm = recency_df.merge(frequency_df, on="CustomerID").merge(monetary_df, on="CustomerID")
st.write("### RFM Data")
st.write(rfm.head())

# Visualize RFM Distributions
fig_rfm = px.scatter_3d(
    rfm,
    x="Recency",
    y="Frequency",
    z="Monetary",
    color="Monetary",
    size="Monetary",
    title="RFM Scatter Plot"
)
st.plotly_chart(fig_rfm)

# K-Means Clustering
st.header("📍 K-Means Clustering")
st.sidebar.subheader("Clustering Parameters")
num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, value=4)

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(rfm[["Recency", "Frequency", "Monetary"]])

# Cluster Visualization
fig_cluster = px.scatter_3d(
    rfm,
    x="Recency",
    y="Frequency",
    z="Monetary",
    color="Cluster",
    title=f"Customer Segmentation with {num_clusters} Clusters",
    symbol="Cluster",
    size="Monetary",
)
st.plotly_chart(fig_cluster)

# Export Data
st.header("📤 Export Processed Data")
if st.button("Export RFM Data"):
    rfm.to_csv("rfm_data.csv", index=False)
    st.success("RFM data exported as `rfm_data.csv`!")

st.markdown("### Enjoy exploring your customer data! 🚀")
