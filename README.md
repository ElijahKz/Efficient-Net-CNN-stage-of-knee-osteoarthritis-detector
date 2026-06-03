
<div style="display: flex; justify-content: center;">
    <img src="https://www.moati.com.au/wp-content/uploads/2022/05/knee-arthritis-surgery.jpg" alt="Descripción" style="width: 100%;">
</div>


Key Features:

1. Intelligent Tuning:

2. Evaluates accuracy after initial training

    * Performs automatic fine-tuning if accuracy <80%

    * Unfreezes only the last layers to prevent overtuning.

3. Integrated Preprocessing:

    * Converts MNIST images (380x380, 1 channel) to the required format (380x380, 3 channels)

    * Normalizes pixel values

4. Complete Serialization:

    * Saves the entire model object (including weights and settings)

    * Can be loaded into another project without additional code

This implementation provides an optimal balance between ease of use and performance, automatically adapting to the data by fine-tuning when necessary.

Advantages of this Design

1. Loose Coupling:

    * Each class has a single responsibility.

1. Easy modification of individual components

2. High cohesion:

    * All evaluation-related code is in the Evaluator

    * All logging logic is in the TrainingMonitor

3. Extensibility:

    * Add new callbacks without modifying the core logic

    * Easy integration of new visualizations

4. Reusability:

    * The TrainingMonitor and Evaluator can be used with other models

    * Model-independent logging system

5. Automatic organization:

    * Consistent directory structure

    * All logs and visualizations are automatically saved
    
![alt text](image.png)


Key Benefits

1. Single Responsibility Principle:

    * TrainingMonitor: Exclusively for callback management and logging

    * Trainer: Only coordinates the training process

    * Evaluator: Only for evaluation and visualization

2. Flexible Configuration:


```
python
# Example: Change monitored metric
monitor = TrainingMonitor()
callbacks = monitor.create_callbacks(monitor_metric='val_accuracy', mode='max')
```



3. Extensibility:


```
python
# Add new callback without modifying existing classes
new_callback = MyCustomCallback()
model.train(..., custom_callbacks=[new_callback])
```
4. Consistency:

    * All experiments follow the same directory structure

    * Centralized callback configuration

    5. Reusability:

    * The monitoring system works with any Keras model

    * The trainer can be used with different architectures

This report includes several key metrics calculated for each class individually and averaged:

a) Precision

    * Formula: True Positives / (True Positives + False Positives)

    * Interpretation: Of all the examples that the model predicted as class X, what percentage were actually class X?

    * Importance: This is crucial when false positives are costly (e.g., incorrect medical diagnosis).

b) Recall/Sensitivity

    * Formula: True Positives / (True Positives + False Negatives)

    * Interpretation: Of all the examples that are actually class X, what percentage did the model correctly identify?

    * Importance: Essential when false negatives are critical (e.g., detecting rare diseases).

c) F1 Score

    * Formula: 2 * (Precision * Recall) / (Precision + Recall)

    * Interpretation: Harmonic mean between precision and recall.

    * Importance: Good balance when you need to consider both false positives (FP and FN).

d) ROC Curves (Receiver Operating Characteristic)

    * Shows the relationship between the True Positive Rate (Recall) and the False Positive Rate.

    * AUC (Area Under the Curve):

    * 1.0: Perfect classifier

    * 0.5: Equivalent to chance

    * In multiclass: A "one-vs-rest" strategy is used.

    * Interpretation: Shows the model's performance at all possible decision thresholds.

e) Precision-Recall Curves

    * Shows the relationship between Precision and Recall for different thresholds.

    * Importance: Especially useful when:

    * Classes are unbalanced

    * We are more interested in the positive class than the negative one

    * AUC: The area under this curve indicates good performance when it is close to 1.0

Learning Curves

    * Shows how training and validation scores evolve as the dataset size increases.

    * Important patterns:

    * Good fit: Both curves converge to a high value

    * Overfit: Large gap between curves (much higher training score)

    * Underfit: Both curves converge to a low value
