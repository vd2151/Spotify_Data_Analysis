import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

n_number = 17603523
# Seeding the random number generator
np.random.seed(n_number)
#%%
#Question 1
spotify_data = pd.read_csv("/Users/vedan/Downloads/spotify52kData.csv")

# Display the first few rows of the dataset to understand its structure
spotify_data.head()


# Selecting the relevant features for analysis
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Creating a 2x5 grid for histograms
plt.figure(figsize=(20, 10))

for i, feature in enumerate(features):
    plt.subplot(2, 5, i + 1)
    sns.histplot(spotify_data[feature], kde=True)
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel('Frequency')


plt.suptitle('Histograms of Song Features')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#%%
#%%
# Question 2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Calculate the correlation coefficient
correlation = spotify_data['duration'].corr(spotify_data['popularity'])

# Create a scatterplot
plt.figure(figsize=(12, 6))
sns.regplot(x='duration', y='popularity', data=spotify_data, scatter_kws={'alpha':0.5})
plt.title('Relationship Between Song Length and Popularity')
plt.xlabel('Duration (ms)')
plt.ylabel('Popularity')
plt.legend([f'Correlation (r): {correlation:.3f}'])
plt.show()
#%%
# Question 3
# Extracting popularity scores for explicit and non-explicit songs
explicit_popularity = spotify_data[spotify_data['explicit'] == True]['popularity']
non_explicit_popularity = spotify_data[spotify_data['explicit'] == False]['popularity']

# Conducting the Mann-Whitney U test to compare the popularity between explicit and non-explicit songs
u_statistic, p_value = mannwhitneyu(explicit_popularity, non_explicit_popularity)
# Boxplot for visual comparison
plt.figure(figsize=(10, 6))
sns.boxplot(x=spotify_data['explicit'], y=spotify_data['popularity'])
plt.title('Popularity of Explicit vs. Non-Explicit Songs')
plt.xlabel('Explicit')
plt.ylabel('Popularity')
plt.show()

u_statistic, p_value

#%%
#Question 4



# Extracting popularity scores
major_key_popularity = spotify_data[spotify_data['mode'] == 1]['popularity']
minor_key_popularity = spotify_data[spotify_data['mode'] == 0]['popularity']

# Mann-Whitney U test
u_statistic_1, p_value_1 = mannwhitneyu(major_key_popularity, minor_key_popularity)
# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='mode', y='popularity', data=spotify_data)
plt.title('Popularity of Songs in Major vs Minor Key')
plt.xlabel('Mode (0 = Minor, 1 = Major)')
plt.ylabel('Popularity')

legend_label = f'Mann-Whitney U: {u_statistic_1:.2f}, p-value_1: {p_value_1:.2e}'
plt.legend([legend_label])

plt.show()
#%%
#Question 5
from scipy.stats import zscore

# Applying z-score normalization to the loudness feature to address skewness
spotify_data['loudness_zscore'] = zscore(spotify_data['loudness'])

# Scatterplot to investigate the relationship between energy and normalized loudness
plt.figure(figsize=(12, 6))
sns.scatterplot(data=spotify_data, x='energy', y='loudness_zscore', alpha=0.5)
plt.title('Relationship Between Energy and Z-Score Normalized Loudness of Songs')
plt.xlabel('Energy')
plt.ylabel('Loudness (Z-Score)')
plt.show()

# Calculating the correlation coefficient between energy and z-score normalized loudness
correlation_energy_loudness_zscore = spotify_data['energy'].corr(spotify_data['loudness_zscore'])
correlation_energy_loudness_zscore

#%%
#question 6

# Calculate correlations
correlations = spotify_data[features].corrwith(spotify_data['popularity'])

# Select the feature with the highest correlation
best_feature = correlations.abs().idxmax()


X = spotify_data[[best_feature]]  # Feature
y = spotify_data['popularity']    # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=n_number)

# Create a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Output the results
print("Best Predictive Feature:", best_feature)
print("Model R²:", r2)
print("Model RMSE:", rmse)
residuals = y_test - y_pred
plt.figure(figsize=(12, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.title('Residuals of Predictions')
plt.xlabel('Predicted Popularity')
plt.ylabel('Residuals')
plt.axhline(y=0, color='k', linestyle='--')
plt.legend([f'R²: {r2:.2f}', f'RMSE: {rmse:.2f}'])
plt.show()


#%%
# Question 7
X = spotify_data[features]  # All features
y = spotify_data['popularity']  # Target variable

# Normalize/Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=n_number)

# Create a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
r21 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)


plt.figure(figsize=(10, 6))

# Scatter plot of actual vs predicted values
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)

# Plotting the regression line
sns.regplot(x=y_test, y=y_pred, scatter=False, color='red')

plt.xlabel('Actual Popularity')
plt.ylabel('Predicted Popularity')
plt.title('Actual vs Predicted Popularity with Regression Line')
plt.show()


# Output the results
print("Multivariate Model R²:", r21)
print("Multivariate Model RMSE:", rmse)
#%%
#Question 8


# Selecting the relevant features
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X = spotify_data[features]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
cumulative_variance = pca.explained_variance_ratio_.cumsum()

# Plotting the cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# Calculating inertia for KMeans clustering to determine the number of clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state= n_number)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

# Plotting the elbow graph for KMeans clustering
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Determining Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

#%%
#Question 9

y_mode = spotify_data['mode']
X_valence = spotify_data[['valence']] 
X_train_valence, X_test_valence, y_train_mode, y_test_mode = train_test_split(X_valence, y_mode, test_size=0.2, random_state=n_number)

# Creating and training the logistic regression model
log_reg_valence = LogisticRegression()
log_reg_valence.fit(X_train_valence, y_train_mode)

# Making predictions on the test set
y_pred_mode_valence = log_reg_valence.predict(X_test_valence)
y_pred_prob_valence = log_reg_valence.predict_proba(X_test_valence)[:, 1]

# Evaluating the model
accuracy_valence = accuracy_score(y_test_mode, y_pred_mode_valence)
fpr, tpr, _ = roc_curve(y_test_mode, y_pred_prob_valence)
roc_auc_valence = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_valence:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Valence Predicting Key')
plt.legend(loc="lower right")
plt.show()

accuracy_valence, roc_auc_valence


#%%
# Question 10

label_encoder = LabelEncoder()
y_genre_encoded = label_encoder.fit_transform(spotify_data['track_genre'])

# Splitting the data for the original features
X_train_features, X_test_features, y_train_genre_encoded, y_test_genre_encoded = train_test_split(X, y_genre_encoded, test_size=0.2, random_state=n_number)

# Creating and training the decision tree classifier with the original features
decision_tree_features = DecisionTreeClassifier(random_state=n_number)
decision_tree_features.fit(X_train_features, y_train_genre_encoded)

# Making predictions and evaluating the model with the original features
y_pred_genre_features = decision_tree_features.predict(X_test_features)
accuracy_genre_features = accuracy_score(y_test_genre_encoded, y_pred_genre_features)

# Splitting the data for PCA components
X_train_pca, X_test_pca, y_train_genre_encoded, y_test_genre_encoded = train_test_split(X_pca, y_genre_encoded, test_size=0.2, random_state= n_number)

# Creating and training the decision tree classifier with PCA components
decision_tree_pca = DecisionTreeClassifier(random_state=n_number)
decision_tree_pca.fit(X_train_pca, y_train_genre_encoded)

# Making predictions and evaluating the model with PCA components
y_pred_genre_pca = decision_tree_pca.predict(X_test_pca)
accuracy_genre_pca = accuracy_score(y_test_genre_encoded, y_pred_genre_pca)

print (accuracy_genre_features)
print(accuracy_genre_pca)

model_accuracies = {
    'Original Features': accuracy_genre_features,
    'PCA Components': accuracy_genre_pca
}

# Plotting the accuracies for visual comparison
plt.figure(figsize=(8, 6))
sns.barplot(x=list(model_accuracies.keys()), y=list(model_accuracies.values()))
plt.title('Comparison of Model Accuracies: Original Features vs. PCA Components')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  
plt.show()
#%%
# Extra credit
grouped_by_time_signature = spotify_data.groupby('time_signature').agg({'popularity':'mean', 'danceability':'mean', 'energy':'mean', 'valence':'mean'}).reset_index()

grouped_by_time_signature

