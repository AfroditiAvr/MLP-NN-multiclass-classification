import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# One-Hot Encoding 
class OneHotEncoder:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def encode(self, labels):
        one_hot = np.zeros((labels.size, self.num_classes))
        one_hot[np.arange(labels.size), labels] = 1
        return one_hot

class Encoder(tf.keras.Model):
    def __init__(self, latent_dim, layer_dims=None):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.layer_dims = layer_dims if layer_dims else [128, latent_dim]
        
        self.layers_list = []
        for i in range(len(self.layer_dims) - 1):
            self.layers_list.append(tf.keras.layers.Dense(self.layer_dims[i], activation='relu'))
        self.latent_layer = tf.keras.layers.Dense(self.latent_dim)
    
    def call(self, inputs):
        img, label = inputs
        x = tf.keras.layers.Flatten()(img)
        x = tf.concat([x, label], axis=1)
        for layer in self.layers_list:
            x = layer(x)
        return self.latent_layer(x)

class Decoder(tf.keras.Model):
    def __init__(self, latent_dim, output_image_shape=(28, 28, 1), layer_dims=None):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_image_shape = output_image_shape
        self.layer_dims = layer_dims if layer_dims else [128, 28 * 28]
        
        self.layers_list = []
        for i in range(len(self.layer_dims) - 1):
            self.layers_list.append(tf.keras.layers.Dense(self.layer_dims[i], activation='relu'))
        self.output_layer = tf.keras.layers.Dense(np.prod(self.output_image_shape), activation='sigmoid')
    
    def call(self, z, label):
        x = tf.concat([z, label], axis=1)
        for layer in self.layers_list:
            x = layer(x)
        output = self.output_layer(x)
        output = tf.reshape(output, (-1, 28, 28, 1))
        return output

class Autoencoder(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def call(self, inputs):
        img, label = inputs
        z = self.encoder([img, label])
        return self.decoder(z, label)

class AutoencoderTrainer:
    def __init__(self, model, batch_size=256, epochs=50, optimizer='adam', loss_fn='binary_crossentropy'):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = tf.keras.optimizers.get(optimizer)
        self.loss_fn = tf.keras.losses.get(loss_fn)
    
    def train(self, X_train, y_train, X_train_next):
        for epoch in range(self.epochs):
            start_time = time.time()
            epoch_loss = 0  # To accumulate loss for the epoch

            for batch_idx in range(0, len(X_train), self.batch_size):
                X_batch = X_train[batch_idx: min(batch_idx + self.batch_size, len(X_train))]
                y_batch = y_train[batch_idx: min(batch_idx + self.batch_size, len(y_train))]
                X_batch_next = X_train_next[batch_idx: min(batch_idx + self.batch_size, len(X_train_next))]

                X_batch_next = np.expand_dims(X_batch_next, axis=-1)  # Add the extra dimension for channels

                with tf.GradientTape() as tape:
                    generated_images = self.model([X_batch, y_batch])

                    generated_images = tf.expand_dims(generated_images, axis=-1) 

                    loss = self.loss_fn(X_batch_next, generated_images)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                # Add the mean loss to epoch_loss
                epoch_loss += tf.reduce_mean(loss).numpy()

            # Average loss for the epoch
            average_loss = epoch_loss / (len(X_train) // self.batch_size + (len(X_train) % self.batch_size > 0))
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time * (self.epochs - epoch - 1)

            # Print progress information
            print(f"Epoch {epoch + 1}/{self.epochs}:")
            print(f"  Reconstruction Error (Loss): {average_loss:.4f}")


class AutoencoderGenerator:
    def __init__(self, model):
        self.model = model
    
    def generate_next_digit(self, input_image, next_digit_label):
        one_hot_encoder = OneHotEncoder(num_classes=10)
        next_digit_onehot = one_hot_encoder.encode(np.array([next_digit_label]))
        generated_image = self.model.predict([input_image, next_digit_onehot])
        return generated_image.reshape(28, 28)


class DigitRecognizer(tf.keras.Model):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def train_digit_recognizer():
    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize
    X_train = X_train[..., np.newaxis]  # Add channel dimension
    X_test = X_test[..., np.newaxis]  # Add channel dimension

    # One-hot encode labels
    one_hot_encoder = OneHotEncoder(num_classes=10)
    y_train_onehot = one_hot_encoder.encode(y_train)
    y_test_onehot = one_hot_encoder.encode(y_test)

    # Initialize and compile the model
    digit_recognizer = DigitRecognizer()
    digit_recognizer.compile(optimizer='adam', 
                             loss='categorical_crossentropy', 
                             metrics=['accuracy'])

    # Train the model
    digit_recognizer.fit(X_train, y_train_onehot, 
                         validation_data=(X_test, y_test_onehot), 
                         epochs=5, 
                         batch_size=256)

    # Evaluate on test data
    test_loss, test_accuracy = digit_recognizer.evaluate(X_test, y_test_onehot, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Generate confusion matrix
    y_pred = digit_recognizer.predict(X_test).argmax(axis=1)  # Get predicted labels
    cm = confusion_matrix(y_test, y_pred)  # Compare true and predicted labels
    
    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    disp.plot(cmap='Blues')  # Set the color map to 'Blues'
    plt.title("Confusion Matrix")
    plt.show()

    # Return the trained model
    return digit_recognizer
        

if __name__ == "__main__":
    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize
    X_train = X_train[..., np.newaxis]  # Add channel dimension
    X_test = X_test[..., np.newaxis]  # Add channel dimension

    # Prepare the "next digit" images
    y_train_next = (y_train + 1) % 10
    y_test_next = (y_test + 1) % 10

    X_train_next, X_test_next = np.zeros_like(X_train), np.zeros_like(X_test)
    for i in range(10):
        indices = np.where(y_train == i)[0]
        next_indices = np.where(y_train == (i + 1) % 10)[0]
        for idx, next_idx in zip(indices, next_indices):
            X_train_next[idx] = X_train[next_idx]

    for i in range(10):
        indices = np.where(y_test == i)[0]
        next_indices = np.where(y_test == (i + 1) % 10)[0]
        for idx, next_idx in zip(indices, next_indices):
            X_test_next[idx] = X_test[next_idx]

    # One-hot encode the labels for the next digits
    one_hot_encoder = OneHotEncoder(num_classes=10)
    y_train_onehot = one_hot_encoder.encode(y_train_next)
    y_test_onehot = one_hot_encoder.encode(y_test_next)

    # Initialize and train the autoencoder
    encoder = Encoder(latent_dim=256)
    decoder = Decoder(latent_dim=256)
    autoencoder = Autoencoder(encoder, decoder)
    trainer = AutoencoderTrainer(autoencoder, batch_size=256, epochs=10)
    trainer.train(X_train, y_train_onehot, X_train_next)

    # Generate next digits
    generator = AutoencoderGenerator(autoencoder)

    # Randomly select 10 different indices
    random_indices = np.random.choice(len(X_test), size=10, replace=False)
    input_images = X_test[random_indices].reshape(10, 28, 28, 1)
    next_digit_labels = y_test_next[random_indices]
    generated_images = [generator.generate_next_digit(input_images[i:i+1], next_digit_labels[i]) for i in range(10)]

    # Display the first image (input digits and generated next digits)
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        axes[0, i].imshow(input_images[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f"Input: {y_test[random_indices[i]]}")
        axes[0, i].axis('off')
        axes[1, i].imshow(generated_images[i], cmap='gray')
        axes[1, i].set_title(f"Generated: {next_digit_labels[i]}")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.show()

    # Train the digit recognizer
    digit_recognizer = train_digit_recognizer()

    # Predict the labels for the generated images
    generated_images = np.array(generated_images).reshape(10, 28, 28, 1)
    predicted_labels = [digit_recognizer.predict(generated_images[i:i+1]).argmax() for i in range(10)]

    # Display the second image (generated digits with predicted labels)
    fig, axes = plt.subplots(1, 10, figsize=(15, 3))
    for i in range(10):
        axes[i].imshow(generated_images[i].squeeze(), cmap='gray')
        axes[i].set_title(f"Pred: {predicted_labels[i]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()