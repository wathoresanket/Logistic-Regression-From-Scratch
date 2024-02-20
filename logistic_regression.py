import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def regress_fit(X_train, y_train, X_test):
    # Normalize the training/testing features using min-max normalization
    X_train_min = np.min(X_train, axis=1, keepdims=True)
    X_train_max = np.max(X_train, axis=1, keepdims=True)
    X_test_min = np.min(X_test, axis=1, keepdims=True)
    X_test_max = np.max(X_test, axis=1, keepdims=True)
    X_train_normalized = (X_train - X_train_min) / (X_train_max - X_train_min)
    X_test_normalized = (X_test - X_test_min) / (X_test_max - X_test_min)

    # Set learning rate and initialize weights
    learning_rate = 1e-3
    num_features = X_train.shape[0]
    weights = np.random.uniform(-1/np.sqrt(num_features), 1/np.sqrt(num_features), (num_features + 1, 1))
    X_train_with_bias = np.ones((num_features + 1, X_train.shape[1]))
    X_train_with_bias[:num_features,:] = X_train_normalized

    # Set number of epochs
    num_epochs = 500
    train_loss_list = []
    train_accuracy_list = []

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0

        for i in range(X_train.shape[1]):
            # Compute raw logits
            raw_logit = np.dot(weights.T, X_train_with_bias[:, i:i+1])[0, 0]
            # Apply sigmoid activation function
            predicted_probability = 1 / (1 + np.exp(-raw_logit))
            actual_label = y_train[i]

            # Compute binary cross-entropy loss
            loss = -(actual_label * np.log(predicted_probability) + (1 - actual_label) * np.log(1 - predicted_probability))
            total_loss += loss

            # Update weights using gradient descent
            gradient = (predicted_probability - actual_label) * X_train_with_bias[:, i:i+1]
            weights -= learning_rate * gradient

            # Check for correct predictions
            if abs(predicted_probability - actual_label) < 0.5:
                correct_predictions += 1

        # Compute average loss and accuracy for the epoch
        average_loss = total_loss / X_train.shape[1]
        accuracy = (correct_predictions / X_train.shape[1]) * 100

        train_loss_list.append(average_loss)
        train_accuracy_list.append(accuracy)

        # Display progress
        progress_message = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {average_loss:.4f} - Train Accuracy: {accuracy:.4f} "
        sys.stdout.write('\r' + progress_message)
        sys.stdout.flush()

    # Plot loss and accuracy curves
    epochs = np.arange(1, num_epochs+1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, train_loss_list, label='Train Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Epochs')

    ax2.plot(epochs, train_accuracy_list, label='Train Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Epochs')

    plt.tight_layout()
    plt.show()

    # Testing
    X_test_with_bias = np.ones((num_features + 1, X_test.shape[1]))
    X_test_with_bias[:num_features,:] = X_test_normalized
    test_logits = np.dot(weights.T, X_test_with_bias)
    predicted_probabilities = 1 / (1 + np.exp(-test_logits))
    y_test_pred = (predicted_probabilities >= 0.5).astype(int)

    return y_test_pred

# Load and fit the model
def load_and_fit():
    df = pd.read_csv("diabetes.csv")
    X = df.drop("Outcome", axis=1)
    X2 = np.array(X).T
    y = df["Outcome"]
    X_train = X2[:,:614]
    X_test = X2[:,614:]
    y_train = y[:614]
    y_test = y[614:]

    # Fit the model
    y_test_pred = regress_fit(X_train, y_train, X_test)

    # Evaluate the accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred[0])
    print(f"Test Accuracy using your implementation: {test_accuracy:.5f}")

    return round(test_accuracy, 5)

load_and_fit()
