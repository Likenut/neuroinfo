import numpy as np

x = np.load(r"C:\Users\magnu\Desktop\project\x.npy")
y = np.load(r"C:\Users\magnu\Desktop\project\y.npy")

# print(data1.shape, data1.dtype)
# print(data2.shape, data2.dtype)

# print(data2)  # Print full content (small arrays)
# print(data1[:5000])  # Print first 5 elements if it's large

import matplotlib.pyplot as plt

# Select a sample trial to visualize
sample_idx = 0  # Change this to see different trials
neural_activity = x[sample_idx]  # Shape: (neurons, time)

plt.figure(figsize=(10, 6))
plt.imshow(neural_activity, aspect="auto", cmap="hot", interpolation="nearest")
plt.colorbar(label="Firing rate")
plt.xlabel("Time")
plt.ylabel("Neuron index")
plt.title(f"Neural Activity for Trial {sample_idx} (Label: {y[sample_idx]})")
plt.show()


# Compute mean firing rate over time for both classes
mean_familiar = np.mean(x[y == 0], axis=0)  # Average across trials with y=0
mean_unfamiliar = np.mean(x[y == 1], axis=0)  # Average across trials with y=1

# Plot the average neural responses
plt.figure(figsize=(10, 5))
plt.plot(np.mean(mean_familiar, axis=0), label="Familiar", color="blue")
plt.plot(np.mean(mean_unfamiliar, axis=0), label="Unfamiliar", color="red")
plt.xlabel("Time")
plt.ylabel("Average Firing Rate")
plt.legend()
plt.title("Average Neural Activity for Familiar vs Unfamiliar Stimuli")
plt.show()



# Compute the average firing rate for each neuron over time
X_features = np.mean(x, axis=2)  # Shape: (800, neurons)


import matplotlib.pyplot as plt
import seaborn as sns

# Split data into two groups
X_familiar = X_features[y == 0].flatten()  # Flatten to compare distributions
X_unfamiliar = X_features[y == 1].flatten()

# Plot the distributions
plt.figure(figsize=(10, 5))
sns.kdeplot(X_familiar, label="Familiar", shade=True, color="blue")
sns.kdeplot(X_unfamiliar, label="Unfamiliar", shade=True, color="red")
plt.xlabel("Average Firing Rate")
plt.ylabel("Density")
plt.title("Distribution of Firing Rates for Familiar vs. Unfamiliar Stimuli")
plt.legend()
plt.show()


from scipy.stats import wasserstein_distance

# Compute the Wasserstein Distance
wd = wasserstein_distance(X_familiar, X_unfamiliar)
print(f"Wasserstein Distance between familiar and unfamiliar firing distributions: {wd:.4f}")

from scipy.stats import entropy

# Estimate probability densities
hist_familiar, bins_familiar = np.histogram(X_familiar, bins=50, density=True)
hist_unfamiliar, bins_unfamiliar = np.histogram(X_unfamiliar, bins=50, density=True)

# Normalize to get probability distributions
hist_familiar += 1e-10  # Avoid zero probabilities
hist_unfamiliar += 1e-10
hist_familiar /= np.sum(hist_familiar)
hist_unfamiliar /= np.sum(hist_unfamiliar)

# Compute KL divergence
kl_div = entropy(hist_familiar, hist_unfamiliar)
print(f"KL Divergence between distributions: {kl_div:.4f}")








from sklearn.model_selection import train_test_split

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42, stratify=y)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)



from sklearn.linear_model import LogisticRegression

# Initialize and train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)




from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions
y_pred = model.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

"""cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Familiar", "Unfamiliar"], yticklabels=["Familiar", "Unfamiliar"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()




# Get logistic regression coefficients
neuron_importance = np.abs(model.coef_).flatten()

# Plot top 10 neurons
top_neurons = np.argsort(neuron_importance)[-10:]
plt.figure(figsize=(10, 5))
plt.bar(range(10), neuron_importance[top_neurons], tick_label=top_neurons)
plt.xlabel("Neuron Index")
plt.ylabel("Importance")
plt.title("Top 10 Most Important Neurons for Classification")
plt.show()"""





from sklearn.metrics import accuracy_score, classification_report, f1_score

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)

# Compute F1-score
f1 = f1_score(y_test, y_pred)

# Print classification report (includes Precision, Recall, and F1-score)
print("\nClassification Report:" + classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Print specific metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
