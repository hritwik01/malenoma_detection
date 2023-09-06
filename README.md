# malenoma_detection
This is a machine learning project for classifying skin cancer types using a convolutional neural network (CNN). Here's a step-by-step explanation of the code:

1. **Creating Training and Validation Datasets**:
   - The code starts by creating training and validation datasets using `tf.keras.preprocessing.image_dataset_from_directory`. This function reads images from directories and converts them into TensorFlow datasets.
   - Two datasets are created, one for training (`train_ds`) and another for validation (`val_ds`). These datasets are split with a validation split of 20%.

2. **Class Names**:
   - The code retrieves the class names from the `train_ds` dataset using the `train_ds.class_names` attribute. These class names correspond to the subdirectories within the training data directory.

3. **Data Visualization**:
   - The code contains a visualization section that displays sample images from the training dataset along with their corresponding class labels. It uses Matplotlib to create a 3x3 grid of images with labels.

4. **Data Preprocessing**:
   - The next section of code is responsible for data preprocessing and augmentation. It uses TensorFlow's `tf.data.experimental.AUTOTUNE` to optimize data loading. Specifically, it caches and prefetches data to improve training efficiency.
   - Data augmentation is applied using the `Augmentor` library to create additional training samples. It includes operations like random rotation, random flipping, and random zooming to introduce variability into the training data.

5. **Model Architecture**:
   - The code defines a CNN model (`model_2`) using TensorFlow's Keras API. This model is designed for image classification and includes convolutional layers, max-pooling layers, a dropout layer, and fully connected (dense) layers.
   - The first layers in the model include data preprocessing steps such as rescaling pixel values to the range [0, 1] and applying the data augmentation defined earlier.
   - The model ends with a dense layer with a number of units equal to the number of classes (9 in this case).

6. **Model Compilation**:
   - The model is compiled with an optimizer (Adam), a loss function (Sparse Categorical Crossentropy), and metrics (accuracy) to prepare it for training.

7. **Model Training**:
   - The code trains the model (`model_2`) using the training and validation datasets. It specifies the number of training epochs (20 in this case) and tracks training history, including training accuracy, validation accuracy, training loss, and validation loss.

8. **Visualizing Training Results**:
   - After training, the code visualizes the training and validation accuracy as well as the training and validation loss over epochs using Matplotlib.

9. **Class Distribution Analysis**:
   - The code analyzes the class distribution in the training dataset. It counts the number of samples for each class before and after data augmentation. This analysis helps address class imbalance issues by adding more samples to minority classes.

10. **Additional Data Augmentation**:
    - The code uses the `Augmentor` library to further augment the data by applying random transformations to create additional samples. This step aims to balance the class distribution.

11. **Creating New Training and Validation Datasets**:
    - New training and validation datasets (`train_ds` and `val_ds`) are created after adding augmented data.

12. **Model Architecture (With Augmentation)**:
    - A new model (`model_3`) is defined with the same architecture as `model_2`, but it includes the additional augmented data in training.

13. **Model Compilation and Training (With Augmentation)**:
    - `model_3` is compiled and trained using the augmented datasets.

14. **Visualizing Training Results (With Augmentation)**:
    - The code visualizes the training and validation accuracy as well as the training and validation loss for `model_3`.

In summary, this code sets up a pipeline for training a CNN model to classify skin cancer types. It includes data preprocessing, augmentation, model creation, training, and evaluation. Additionally, it addresses class imbalance by augmenting data for minority classes. The code provides insights into the training progress and helps in fine-tuning the model for better performance.
