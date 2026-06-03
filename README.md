
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
    
![image.png](data:image/png;base64,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)

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
