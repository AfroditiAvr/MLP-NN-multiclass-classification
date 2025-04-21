import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load CIFAR-10 dataset from TensorFlow
cifar10 = tf.keras.datasets.cifar10
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Normalize and reshape data
X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0  # Flatten and normalize
X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0     # Flatten and normalize

Y_train = Y_train.flatten()  # Flatten labels to 1D array
Y_test = Y_test.flatten()    # Flatten labels to 1D array

# Debugging shape information
print(f"Training set shape: {X_train.shape}, Labels: {Y_train.shape}")
print(f"Test set shape: {X_test.shape}, Labels: {Y_test.shape}")

def pca(X, num_components):
    # Step 1: Center the data
    mean_vector = np.mean(X, axis=1, keepdims=True)  # Compute mean for each feature
    X_centered = X - mean_vector  # Center the data

    # Step 2: Compute covariance matrix
    covariance_matrix = np.cov(X_centered)

    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 4: Sort eigenvectors by descending eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_indices[:num_components]]  # Select top components

    # Step 5: Project data onto the top principal components
    X_reduced = top_eigenvectors.T @ X_centered  # Transform data

    return X_reduced, top_eigenvectors, mean_vector

# Apply PCA to reduce dimensions to 300
num_components = 300
X_train_pca, pca_components, train_mean_vector = pca(X_train, num_components=num_components)

# Center the test data using the mean of the training data
X_test_centered = X_test - train_mean_vector  # Center test data
X_test_pca = pca_components.T @ X_test_centered  # Project the test data

# Debugging shape information
print(f"Training set shape after PCA: {X_train_pca.shape}, Labels: {Y_train.shape}")
print(f"Test set shape after PCA: {X_test_pca.shape}, Labels: {Y_test.shape}")

# Neural network functions
def init_params(input_size=300):
    W1 = np.random.rand(50, input_size) - 0.5 
    b1 = np.random.rand(50, 1) - 0.5
    W2 = np.random.rand(10, 50) - 0.5   # Hidden layer size set to 50
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y, num_classes=10):
    """Converts an array of labels into one-hot encoding."""
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def compute_loss(A2, Y):
    one_hot_Y = one_hot(Y, num_classes=10)
    m = Y.size
    loss = -np.sum(one_hot_Y * np.log(A2 + 1e-8)) / m  # Add epsilon for numerical stability
    return loss

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y, num_classes=10)
    m = X.shape[1]

    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def create_mini_batches(X, Y, batch_size):
    """Split the data into mini-batches."""
    m = X.shape[1]
    indices = np.random.permutation(m)
    X_shuffled = X[:, indices]
    Y_shuffled = Y[indices]

    mini_batches = []
    for k in range(0, m, batch_size):
        X_batch = X_shuffled[:, k:k + batch_size]
        Y_batch = Y_shuffled[k:k + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches


def gradient_descent(X_train_pca, Y_train, X_test_pca, Y_test, alpha, epochs, batch_size):
    W1, b1, W2, b2 = init_params(input_size=X_train_pca.shape[0])
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies= []

    for i in range(epochs):
        mini_batches = create_mini_batches(X_train_pca, Y_train, batch_size)
        epoch_loss = 0
        for X_batch, Y_batch in mini_batches:
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_batch)
            loss = compute_loss(A2, Y_batch)
            epoch_loss += loss

            # Backward propagation and parameter updates
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X_batch, Y_batch)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

            

        # Average training loss over all mini-batches
        epoch_loss /= len(mini_batches)
        train_losses.append(epoch_loss)

        # Calculate training accuracy and loss on PCA-transformed training data
        _, _, _, A2_train = forward_prop(W1, b1, W2, b2, X_train_pca)
        train_predictions = get_predictions(A2_train)
        train_accuracy = get_accuracy(train_predictions, Y_train)
        train_accuracies.append(train_accuracy)
        train_loss = compute_loss(A2_train, Y_train)

        # Calculate test accuracy and loss on PCA-transformed test data
        _, _, _, A2_test = forward_prop(W1, b1, W2, b2, X_test_pca)
        test_predictions = get_predictions(A2_test)
        test_accuracy = get_accuracy(test_predictions, Y_test)
        test_accuracies.append(test_accuracy)
        test_loss = compute_loss(A2_test, Y_test)

        test_losses.append(test_loss)

        # Print progress every 20 epochs or at the last epoch
        if (i + 1) % 20 == 0 or i == epochs - 1:
            print(f"Epoch {i + 1}/{epochs}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, "
                  f"Train Accuracy = {train_accuracy:.4f}, Test Accuracy = {test_accuracy:.4f}")

    # Plot training and testing accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Training and Test Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot loss curves for both training and test datasets
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='orange')
    plt.title('Training and Test Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    return W1, b1, W2, b2, test_predictions

# Train the neural network
W1, b1, W2, b2, test_predictions = gradient_descent(X_train_pca, Y_train, X_test_pca, Y_test, alpha=0.01, epochs=500, batch_size=64)

# Create the confusion matrix
cm = confusion_matrix(Y_test, test_predictions)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))  # CIFAR-10 labels
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Prediction functions
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# Function to get representative images for predicted labels
def get_predicted_images(predictions):
    """
    For each predicted label, randomly select an image from the test set that belongs to the same class.
    """
    predicted_images = []
    for pred in predictions:
        # Find all indices in the test set with the predicted label
        indices = np.where(Y_test == pred)[0]
        
        # Randomly select one such image
        chosen_index = random.choice(indices)
        predicted_image = X_test[:, chosen_index].reshape((32, 32, 3)) * 255  # Un-normalize
        
        predicted_images.append(predicted_image.astype(np.uint8))
    return np.array(predicted_images)

#test predictions and return true and predicted images
def test_predictions_with_images(positions, W1, b1, W2, b2):
    true_images = []  # True test images
    predictions = []  # Predicted labels
    true_labels = []  # True labels

    for index in positions:
        # Get the original test image and its label
        current_image = X_test[:, index].reshape(-1, 1)
        label = Y_test[index]

        # Project the test image onto PCA space
        current_image_centered = current_image - train_mean_vector
        current_image_pca = pca_components.T @ current_image_centered

        # Make predictions
        prediction = make_predictions(current_image_pca, W1, b1, W2, b2)

        # Collect true and predicted information
        predictions.append(prediction[0])
        true_labels.append(label)

        # Reshape the true image for visualization
        reshaped_image = current_image.reshape((32, 32, 3)) * 255
        true_images.append(reshaped_image.astype(np.uint8))

    # Get representative images for predicted labels
    predicted_images = get_predicted_images(predictions)

    return np.array(true_images), np.array(predicted_images), true_labels, predictions

# Select random test set positions for visualization
num_positions = 10
positions = random.sample(range(X_test.shape[1]), num_positions)

# Get true and predicted images along with labels
true_images, predicted_images, true_labels, predictions = test_predictions_with_images(positions, W1, b1, W2, b2)

# Display true images and predicted representative images
plt.figure(figsize=(15, 6))

for i, (true_img, pred_img, true_label, pred_label) in enumerate(zip(true_images, predicted_images, true_labels, predictions)):
    # True images
    plt.subplot(2, num_positions, i + 1)
    plt.imshow(true_img)
    plt.title(f"True: {true_label}")
    plt.axis('off')

    # Predicted images
    plt.subplot(2, num_positions, i + 1 + num_positions)
    plt.imshow(pred_img)
    plt.title(f"Pred: {pred_label}")
    plt.axis('off')

    # Print the results in the terminal
    print(f"Image {i + 1}:")
    print(f"    True label: {true_label}")
    print(f"    Predicted label: {pred_label}")
    print("-" * 30)

plt.tight_layout()
plt.show()
