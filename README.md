import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# 1. Load & check
df = pd.read_csv("Mall_Customers.csv")
print(df.head())
df.info()
print("Missing:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# 2. EDA
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for a, col in zip(ax, features):
    sns.histplot(df[col], kde=True, ax=a)
    a.set_title(col)
plt.tight_layout()
plt.show()

# 3. Preprocessing
df['Gender_Code'] = df['Gender'].map({'Male': 0, 'Female': 1})
X_scaled = StandardScaler().fit_transform(df[features])

# 4. Elbow + Silhouette
inertia, sil = [], []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(K_range, inertia, marker='o')
ax[0].set_title('Elbow Method')
ax[0].set_xlabel('K')
ax[1].plot(K_range, sil, marker='o', color='green')
ax[1].set_title('Silhouette Score')
ax[1].set_xlabel('K')
plt.show()

# 5. Final KMeans
N_CLUSTERS = 5
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print("Silhouette:", round(silhouette_score(X_scaled, df['Cluster']), 3))

# 6. Segment profile
summary = df.groupby('Cluster')[features].mean().round(1)
summary['Count'] = df['Cluster'].value_counts().sort_index()
summary['Female %'] = (df.groupby('Cluster')['Gender_Code'].mean() * 100).round(1)

# 7. Auto segment names
inc_mid = df['Annual Income (k$)'].mean()
spd_mid = df['Spending Score (1-100)'].mean()

def name_segment(r):
    inc, spd = r['Annual Income (k$)'], r['Spending Score (1-100)']
    if inc >= inc_mid + 10 and spd >= spd_mid + 10: return 'Premium Customers'
    if inc >= inc_mid + 10 and spd < spd_mid - 10:  return 'Potential Customers'
    if inc < inc_mid - 10 and spd >= spd_mid + 10:  return 'Impulsive Buyers'
    if inc < inc_mid - 10 and spd < spd_mid - 10:   return 'Budget Customers'
    return 'Regular Customers'

actions = {
    'Premium Customers': 'Loyalty rewards, VIP membership',
    'Potential Customers': 'Personalized promotions',
    'Impulsive Buyers': 'Limited-time offers, flash sales',
    'Budget Customers': 'Discount coupons',
    'Regular Customers': 'Retention and referral programs',
}

seg_names = summary.apply(name_segment, axis=1)
dup = seg_names.duplicated(keep=False)
seg_names = seg_names.where(~dup, seg_names + ' (C' + seg_names.index.astype(str) + ')')
df['Segment'] = df['Cluster'].map(seg_names)

report = summary.copy()
report['Segment'] = seg_names
report['Marketing Action'] = [actions[n.split(' (C')[0]] for n in seg_names]
print(report[['Segment'] + features + ['Count', 'Female %', 'Marketing Action']])

# 8. Visuals
order = list(seg_names.sort_index().values)
palette = dict(zip(order, sns.color_palette('tab10', len(order))))

plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Segment', palette=palette, s=80)
centers = StandardScaler().fit(df[features]).inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 1], centers[:, 2], c='black', marker='X', s=250, label='Centroids')
plt.title('Customer Segments: Income vs Spending')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
counts = df['Segment'].value_counts()
ax[0].bar(count
