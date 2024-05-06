import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA

# Read the CSV file into a pandas DataFrame
concat_df = pd.read_csv(r"C:\Users\brahm\OneDrive\Desktop\Codes\clean_concat.csv")

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the labels
concat_df['Label'] = label_encoder.fit_transform(concat_df['Label'])

# Suppose your features are stored in X and labels in y
X = concat_df.drop(columns=['Label'])  # Assuming 'Label' is the target column
y = concat_df['Label']

# Scale the features
scaler = MinMaxScaler()
scaledData = scaler.fit_transform(X)

# Instantiate PCA without specifying the number of components
pca = PCA(n_components=None)

# Fit PCA on the scaled data
pca.fit(scaledData)

# Analyze the explained variance ratio to decide the number of components
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Plot the explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance Ratio by Number of Components')
plt.grid(True)

# Save the plot as an image file
plt.savefig("explained_variance_ratio_plot.png")

# Save the DataFrame of explained variance ratio to a CSV file
explained_variance_df = pd.DataFrame({'Explained Variance Ratio': explained_variance_ratio})
explained_variance_df.to_csv("explained_variance_ratio.csv", index=False)

# Determine the number of components based on the threshold of explained variance ratio (e.g., 0.95)
desired_variance_ratio = 0.95
n_components = np.argmax(cumulative_variance_ratio >= desired_variance_ratio) + 1

# Perform PCA again with the chosen number of components
pca_final = PCA(n_components=n_components)
pca_data = pca_final.fit_transform(scaledData)

# Save the PCA-transformed data to a CSV file
pca_data_df = pd.DataFrame(pca_data, columns=[f'PC{i}' for i in range(1, n_components + 1)])
pca_data_df.to_csv("pca_transformed_data.csv", index=False)
