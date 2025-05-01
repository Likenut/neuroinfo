import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, entropy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
from scipy.fftpack import fft
from sklearn.decomposition import PCA


# Loading data; change this to your path
x = np.load(r"C:\Users\Admin\Desktop\neuroinfo\x.npy")
y = np.load(r"C:\Users\Admin\Desktop\neuroinfo\y.npy")


# Select a sample trial to visualize

# To plot the firing rate of neurons over time for a few trials
def sample_trial(x, y, idx = 0):
    sample_idx = idx  # Change this to see different trials
    neural_activity = x[sample_idx]  # Shape: (neurons, time)

    # Plotting sample
    plt.figure(figsize=(10, 6))
    plt.imshow(neural_activity, aspect="auto", cmap="hot", interpolation="nearest")
    plt.colorbar(label="Firing rate")
    plt.xlabel("Time")
    plt.ylabel("Neuron index")
    plt.title(f"Neural Activity for Trial {sample_idx} (Label: {y[sample_idx]})")
    plt.show()


# To check if familiar and unfamiliar stimuli produce different neural responses. 
def firing_rate(x, y, firing_rate_mean = True): # (T = mean or F = sum)

    if firing_rate_mean == True:
        # Compute mean firing rate over time for both classes
        mean_familiar = np.mean(x[y == 0], axis=0)  # Average across trials with y=0
        mean_unfamiliar = np.mean(x[y == 1], axis=0)  # Average across trials with y=1
    else:
        # Compute sum of firing rate over time for both classes
        sum_familiar = np.sum(x[y == 0], axis=0)  # Average across trials with y=0
        sum_unfamiliar = np.sum(x[y == 1], axis=0)  # Average across trials with y=1

    if firing_rate_mean == True:
        # Plot the average neural responses
        plt.figure(figsize=(10, 5))
        plt.plot(np.mean(mean_familiar, axis=0), label="Familiar", color="blue")
        plt.plot(np.mean(mean_unfamiliar, axis=0), label="Unfamiliar", color="red")
        plt.xlabel("Time")
        plt.ylabel("Average Firing Rate")
        plt.legend()
        plt.title("Average Neural Activity for Familiar vs Unfamiliar Stimuli")
        plt.show()
    else:
        # Plot the sum of neural responses
        plt.figure(figsize=(10, 5))
        plt.plot(np.sum(sum_familiar, axis=0), label="Familiar", color="blue")
        plt.plot(np.sum(sum_unfamiliar, axis=0), label="Unfamiliar", color="red")
        plt.xlabel("Time")
        plt.ylabel("Sum of Firing Rate")
        plt.legend()
        plt.title("Sum of Neural Activity for Familiar vs Unfamiliar Stimuli")
        plt.show()


# ------------------------------------------------------------------------


def mean_x(x):
    # Compute the average firing rate for each neuron over time
    X_features = np.mean(x, axis=2)  # Shape: (800, neurons)
    return X_features


def sum_x(x):
    # Instead of tracking individual neurons, sum the activity of all neurons per time step.
    X_features = np.sum(x, axis=1) 
    return X_features


def fourier_x(x):
    # Compute Fourier Transform for each neuron
    X_fft = np.abs(fft(x, axis=2))  # Shape: (800, neurons, frequencies)
    X_features = np.mean(X_fft, axis=2)  # Take mean power across frequencies
    return X_features


def pca_x(x):
    # Flatten x to (800, neurons*time)
    X_flat = x.reshape(x.shape[0], -1)

    # Apply PCA
    pca = PCA(n_components=50)  # Keep 50 components
    X_features = pca.fit_transform(X_flat)
    return X_features


def decide_x(decision = 0): 
    match decision:
        case 0:
            X_features = mean_x(x)
        case 1:
            X_features = sum_x(x)
        case 2:
            X_features = fourier_x(x)
        case 3:
            X_features = pca_x(x)
    
    return X_features

X_features = decide_x(1) # selecting: 0 = mean, 1 = sum, 2 = fourier, 3 = pca


# ------------------------------------------------------------------------


# Split data into two groups and flatten to compare distributions
X_familiar = X_features[y == 0].flatten()  
X_unfamiliar = X_features[y == 1].flatten()


def exploratory_analysis(X_familiar, X_unfamiliar):
    # Plot the distributions
    plt.figure(figsize=(10, 5))
    sns.kdeplot(X_familiar, label="Familiar", fill=True, color="blue")
    sns.kdeplot(X_unfamiliar, label="Unfamiliar", fill=True, color="red")
    plt.xlabel("Average Firing Rate")
    plt.ylabel("Density")
    plt.title("Distribution of Firing Rates for Familiar vs. Unfamiliar Stimuli")
    plt.legend()
    plt.show()


def wasserstein(X_familiar, X_unfamiliar):
    # Compute the Wasserstein Distance
    wd = wasserstein_distance(X_familiar, X_unfamiliar)
    print(f"Wasserstein Distance between familiar and unfamiliar firing distributions: {wd:.4f}")


def kl_divergence(X_familiar, X_unfamiliar):
    # Estimate probability densities
    hist_familiar, bins_familiar = np.histogram(X_familiar, bins=50, density=True) # Bins are needed for some reason
    hist_unfamiliar, bins_unfamiliar = np.histogram(X_unfamiliar, bins=50, density=True)

    # Normalize to get probability distributions
    hist_familiar += 1e-10  # Avoid zero probabilities
    hist_unfamiliar += 1e-10
    hist_familiar /= np.sum(hist_familiar)
    hist_unfamiliar /= np.sum(hist_unfamiliar)

    # Compute KL divergence
    kl_div = entropy(hist_familiar, hist_unfamiliar)
    print(f"KL Divergence between distributions: {kl_div:.4f}")


# -----------------------------------------------------------


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42, stratify=y)


# Initialize and train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


def confusion(y_test, y_pred):
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Familiar", "Unfamiliar"], yticklabels=["Familiar", "Unfamiliar"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


def feature_importance():
    # Feature Importance

    # Get logistic regression coefficients
    neuron_importance = np.abs(model.coef_).flatten()

    # Plot top 10 neurons
    top_neurons = np.argsort(neuron_importance)[-10:]
    plt.figure(figsize=(10, 5))
    plt.bar(range(10), neuron_importance[top_neurons], tick_label=top_neurons)
    plt.xlabel("Neuron Index")
    plt.ylabel("Importance")
    plt.title("Top 10 Most Important Neurons for Classification")
    plt.show()


# -------------------------------------------


def performance(y_test, y_pred):
    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Compute F1-score, precison and recall
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Print classification report (includes Precision, Recall, and F1-score)
    print("\nClassification Report: \n" + classification_report(y_test, y_pred))

    # Print specific metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")


# -----------------------------


sample_trial(x, y, 1) # Task 1

firing_rate(x, y, True) # Task 1

exploratory_analysis(X_familiar, X_unfamiliar) # Task 3

wasserstein(X_familiar, X_unfamiliar) # Task 3

kl_divergence(X_familiar, X_unfamiliar) # Task 3

confusion(y_test, y_pred) # Task 4

feature_importance() # Task 4

performance(y_test, y_pred) # Task 5
