
# Problem Definition:
## Objective: For any input image, classify the input into one of the following classes:
| Image_code | Class | Expected Output |
| :---:   | :---: | :---: |
| 3048 | Character1 | [1.0, 0.0, 0.0] |
| 6775 | Character2 | [0.0, 1.0, 0.0] |
| 12233 | Character3	| [0.0, 0.0, 1.0] |
| - | No of the above classes	| [0.0, 0.0, 0.0] |


## Dataset: 
The training dataset comprises 92,350 .jpeg images stored in the images.tar file.

The images vary in size, necessitating resizing for consistency.

The dataset is unlabelled, meaning similar images are not grouped under specific classes.

Embeddings of the images in the images.tar file are available in embeddings.npy, where each image is represented by a 512-dimensional vector.

The test set (test_set.csv) consists of 100 image indices from the training set, chosen to evaluate the final classification model.

## Data Preprocessing:
•	Find similar images to the requested classes. Here, I applied two different techniques:

1.	Extracting features of images using a pretrained neural network model (ResNet). I removed its top layer, froze pre-trained layers, and added a custom head. Then, I defined transformations for input images and a function to extract features from every image. Finally, I calculated the cosine similarity based on extracted features from the images.
2.	Applying the Cosine Similarity method on the image embedding data. 

Note: I noticed that the similarity reported for selected similar images was almost identical. Therefore, I proceeded with the Cosine Similarity on embedding data because this method was faster compared to the previous method (pretrained model).
![image](https://github.com/yasi44/Unsupervised_Image_Classification/assets/12167377/f3c1453c-9754-44bb-b44c-235cf8094a0f)

•	Resizing images: Since image sizes differ, I resized them while maintaining the scale ratio.

•	Dealing with imbalanced data: Since the number of images in each class varies, techniques were applied to prevent bias towards classes with more elements. The following techniques were implemented:

o	Augmentation (Generating Synthetic Samples): To increase the size of the training dataset, various transformations were applied to existing images. This augmentation was applied before any data splitting. Augmented images were only added to the training set, not the validation or test sets. The function perform_augmentation() was used.

o	Class Weighting: Assigning different weights to different classes during the training phase to address class imbalance. Higher weights were assigned to minority classes, and lower weights to majority classes. This gave more importance to the minority class samples during the training process, effectively balancing their influence on the model's learning process. The function calculate_class_weights() was utilized. 

o	Note 1: Class Weighting was found to be faster, so it was chosen over Augmentation. However, the source code for augmentation is also included.

o	Note 2: Class Weighting was faster. So I moved forward by Class Weighting, but the source code of augmentation is also included.

•	Splitting the dataset into training, validation.  data_preparation()

•	Labeling data: the grouped data assigned to 3+1 label. 3 label for each of "Character1", "Character2", "Character3" classes and the forth one labelled as “etc” to be used for images that belong to neither of those 3 classes.

## Model Selection and Training:
•	A pretrained model ResNet50 was selected as the based classification model.

•	On top of that GlobalAveragePooling2D and Dense layers was added, each followed by a BatchNormalization(to improve gradient flow) and a Dropout layer(to prevent overfitting). 

•	Hyperopt was use for parameter tunning. The most optimum values reported were: learning_rate = 0.0001 , units = 128, dropout = 0.3

•	Similar images clustered based on cosine similarity score on 3 different values: 0.7, 0.75, 0.8. results are listed in the table.

•	Model retrained on the train data and validated on the validation dataset for all 4 different groups using appropriate optimization techniques and hyperparameters.

•	Change the pretrained model from ResNet50 to EfficientNetB0 didn’t increased the accuracy.

•	Since the number of images in training data is not high, I decided to not using many layers, to prevent overfitting.

## Model Evaluation:
•	Evaluated the trained model on the test_set.csv dataset to assess its performance.

•	Since the no True Positive value existed in the test_set.csv, the result is shown as nan for that class.

•	Here we use Multi-class ROC curve that visualizes true positive rate (sensitivity) against the false positive rate (1 - specificity) for different threshold values, allowing us to visualize the trade-off between sensitivity and specificity. It considers each class as the positive class, while treating all other classes as the negative class.

•	Early stopping used to prevent overfilling 
The result of testing the test data on different models:
The optimal values found through Hyperopt fine-tunning: learning_rate = 0.0001 , units = 128, dropout = 0.3
![image](https://github.com/yasi44/Image_Classification_Unsupervised/assets/12167377/2d3f69ca-2a6d-4573-a9ba-43cb16b3976a)

## Result Interpretation and Discussion:
•	Accuracy is higher for both class 0 and class 1 ("Character1" and "Character2").

•	Overall threshold 0.7 has shown more success in clustering similar images, specifically for class 1

•	The accuracy can be improved by improving the cosine similarity and labeling the data, this process can later on be improved by active learning, which can overcome imbalance data problem.

## Future Works:
Some other techniques like Zero-shot learning also can be used with some other information about images of target classes to help it generalize to unseen data. leverages pre-trained models that have been trained on large-scale datasets to extract visual features from the input data. These pre-trained models are fine-tuned or adapted to the zero-shot learning task, allowing them to capture relevant visual information even for unseen classes.
