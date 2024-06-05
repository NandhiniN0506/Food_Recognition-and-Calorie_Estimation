**Food Classification using Deep Learning**
**Overview**

This project aims to classify various food items using deep learning techniques. The dataset used for training and testing is the Food-101 dataset obtained from Kaggle. The dataset consists of 101 food categories with images of each food item.

**Classes and Calories**

Each food item in the dataset is associated with a specific calorie value per gram. These values are approximations and can vary based on factors such as ingredients and cooking methods.

**Dataset**

The Food-101 dataset contains images of various food items. It is downloaded from Kaggle using the Kaggle API. The dataset is then split into training and testing sets.
 [DataSet link:](https://www.kaggle.com/dansbecker/food-101)

**Model Architecture**

The model architecture used for this project is based on the DenseNet201 pre-trained model. The pre-trained weights are used to initialize the model, followed by adding additional layers for classification.

**DenseNet201** specifically refers to a variant of DenseNet that consists of 201 layers. It is a deeper and more complex model compared to its predecessors like DenseNet121 and DenseNet169. DenseNet201 has been pre-trained on large-scale image datasets, such as ImageNet, and has shown impressive performance across various computer vision tasks, including image classification, object detection, and segmentation.

![DenseNet201-architecture-with-extra-layers-added-at-the-end-for-fine-tuning-on-UCF-101](https://github.com/NandhiniN0506/PRODIGY_ML_05/assets/157806111/b8e70a52-934a-476d-b9a1-e906018f80cb)

**Training**

The model is trained using the training data with data augmentation techniques such as rescaling and validation split.Early stopping is employed to prevent overfitting. 

**Early stopping** is a technique used during the training of machine learning models to prevent overfitting. It involves monitoring a chosen metric, such as validation loss or accuracy, and stopping the training process when the metric stops improving or starts deteriorating on a held-out validation dataset. This prevents the model from becoming overly specialized to the training data and helps in achieving better generalization performance on unseen data.

![1_ohJrSjXe-7x_ZMyeIT-AsQ](https://github.com/NandhiniN0506/PRODIGY_ML_05/assets/157806111/b9bec859-91b7-4c33-b5f4-adcc56276e83)

**Testing**

The trained model is evaluated using the testing data to measure its accuracy and loss. The accuracy achieved on the test set reflects the model's performance in classifying unseen food images.


### Sample Predictions

To demonstrate the model's predictions, sample images of different food items are provided. Each image is fed into the model, and the predicted class along with its associated calorie information is displayed.

#### Predictions:

1. Macarons:
   - Class: 52 (macarons)
   - Calories: ~4 calories per gram

2. French Fries:
   - Class: 44 (french_fries)
   - Calories: ~3.5 calories per gram

3. Ice Cream:
   - Class: 46 (ice_cream)
   - Calories: ~2 calories per gram

4. Pizza:
   - Class: 75 (pizza)
   - Calories: ~2.5 calories per gram

5. Donuts:
   - Class: 27 (donuts)
   - Calories: ~4 calories per gram

6. French Toast:
   - Class: 29 (french_toast)
   - Calories: ~2 calories per gram

These predictions are based on the trained model's output for each respective food item.

## Dependencies
- Keras
- TensorFlow
- Matplotlib
- Pandas
- Seaborn
- NumPy
- Scikit-learn

## Usage
To run the code, ensure all dependencies are installed and the Kaggle API is properly configured. Then, execute the code provided in the notebook/script.

## Note
The provided calorie values are approximate and may vary based on ingredients and cooking methods.
