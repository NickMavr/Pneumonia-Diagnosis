Deep Learning for Pneumonia Diagnosis

Project Overview
This project applies deep learning techniques to classify chest X-ray images into three categories:
Normal (Healthy)
Bacterial Pneumonia
Viral Pneumonia

Using advanced convolutional neural network (CNN) architectures, the aim is to enhance diagnostic accuracy and support clinical decision-making. Models like EfficientNet, EfficientNetV2, and MobileNet were utilized and fine-tuned for this task.

Key Features
Preprocessing techniques to handle class imbalance using data augmentation.
Training multiple CNN architectures with hyperparameter tuning.
Ensemble learning using soft voting to improve accuracy.
Evaluation of models based on test data and Kaggle competition results.

Results
The project achieved a notable classification accuracy of 84.2% on the test dataset using an ensemble of models, demonstrating the effectiveness of soft voting techniques over individual classifiers.

Repository Contents
project_code.py: Python file containing the implementation of the models and preprocessing pipeline.
Deep Learning for Pneumonia Diagnosis.pdf: Comprehensive report detailing the models, experimental results, and insights.
README.md: Overview of the project 

Dataset
The dataset comprises chest X-ray images categorized into the three target classes. Due to imbalances, data augmentation was applied to enhance diversity and improve model generalization. For detailed preprocessing steps, refer to the PDF report.

Models and Techniques

Models
EfficientNet (B1, B3): Pretrained on ImageNet; optimized with Swish activation and batch normalization.
EfficientNetV2: Enhanced architecture for better accuracy and efficiency.
MobileNet: Lightweight model suitable for deployment on resource-constrained devices.

Key Training Strategies
Early Stopping: Monitoring validation loss with a patience of 13 epochs.
Model Checkpoints: Saving weights of the best-performing epochs.
Data Augmentation: Rotation, flipping, and scaling to improve robustness.
Ensemble Learning: Soft voting to combine predictions from multiple models.

Future Work
Experiment with additional ensemble methods like bagging and boosting.
Explore model optimization for real-time deployment.
Extend the analysis to include other types of pneumonia or medical imaging modalities.

Citation
If you use this project or its findings, please cite the report or repository as appropriate.

Acknowledgments
Deep Learning Frameworks: TensorFlow, Keras
Pretrained Models: ImageNet
Special thanks to collaborators and data providers.
