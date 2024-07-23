## Malaria Detection using Deep Learning

The objective of this Project is to show how deep learning architecture such as convolutional neural network (CNN) which can be useful in real-time malaria detection effectively and accurately from input images and to reduce manual labor with a web application.

### Task Description
----
The goal of this project is to develop deep learning models that can accurately classify images of blood smears as either infected with malaria parasites (positive) or uninfected (negative). By automating the classification process, the model aims to assist healthcare professionals in quickly and accurately diagnosing malaria, particularly in areas where access to trained personnel and laboratory facilities is limited.

### About Dataset  
----
The dataset contains 2 folders
Infected
Uninfected
And a total of 27,558 images of which 13,780 are infected and 13,780 are not infected. Dataset images are not all the same size.

### Steps
----
The steps involved to build a model are:
1. **Data Loading**: Gather a dataset of images containing blood smear samples and loading the images of cells.
2. **Data Preprocessing**: Resize the images to a uniform size of 244x244 for training.Normalize the pixel values to a range between 0 and 1.Split the dataset into training, validation, and test sets.
3. **Data Augmentation**: I have used the ImageDataGenerator class in Keras to preprocess and augment image data during training. It allows us to perform various transformations on images such as rescaling, rotation, shifting, shearing, zooming, and flipping, among others.
4. **Model Selection**: Select the model having highest accuracy.<br>
    -  **Convolutional Neural Network (CNN)**<br>
       I designed a CNN architecture with multiple convolutional and pooling layers followed by fully connected layers and an output layer.The model was trained using the training dataset and validated using the validation dataset.I used binary cross-entropy loss and Adam optimizer for model training.

    -  **VGG16**<br>
      I have utilized the pre-trained VGG16 model, pre-trained on the ImageNet dataset, for transfer learning and fine-tuned the last few layers of the VGG16 model on malaria cell classification task.The model was trained using the training dataset and validated using the validation dataset.

    -  **ResNet**<br>
  I employed the pre-trained ResNet50 model, pre-trained on the ImageNet dataset, for transfer learning and  fine-tuned the last few layers of the ResNet50 model on malaria cell classification task.The model was 
  trained using the training dataset and validated using the validation dataset.<br>
5. **Model Compilation**:  Compile the model with loss function binary cross-entropy, optimizer Adam and evaluation metrics accuracy.
6. **Model Training**:  Train the model on the training dataset using the fit() function.Monitored the training process to ensure convergence and prevent overfitting.Validated the model's performance on the validation dataset during training.
7. **Model Evaluation**:  Evaluated the trained model's performance on the test dataset using evaluation metrics of accuracy, precision, recall, and F1 score.
Visualize the model's performance metrics using plots between loss and accuracy of training data and validation data.

### Results
----
VGG15 model output
![VGG15 model output](https://github.com/bhavesa16/Deep-Learning-project/blob/main/vgg_output.png)

CNN model output
![cnn output](https://github.com/bhavesa16/Deep-Learning-project/blob/main/cnn_model_op.png)

Resnet model output
![resnet](https://github.com/bhavesa16/Deep-Learning-project/blob/main/resnet_output.png)

Vgg16 model has the highest accuracy among the three models.
### Dataset link
https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria/data

