import warnings
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from data_preprocessing import train_generator,test_generator
import matplotlib.pyplot as plt
# model building
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Binary classification
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model training

# Accuracy Graph
def accuraacy_graph():
       
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.legend()
    plt.show()

#Loss Graph
def loss_graph():
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


key=True



if key:
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='accuracy',  # Monitor accuracy
                                   patience=0,          # Stop as soon as accuracy stops improving
                                   restore_best_weights=True)  # Restore the best weights

    # Train the model with EarlyStopping callback
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=3,
        callbacks=[early_stopping]  # Add EarlyStopping callback
    )

    model.save('caries_gingivitis_model.keras')
    accuraacy_graph()
    loss_graph()
    

# Ensure the model is not saved with low accuracy if training is interrupted  
else:
    print("\n\nSet 'key' to True to train the model.\n\n")  
