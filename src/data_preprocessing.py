#Importing the Required Libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#1.train_data
train_data_path = '../data/Train'

#2.test_data
test_data_path = '../data/Test'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


#here i do all required image preprocessing (only in training_data) which can improve the accuraacy and generalization of the model
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

#building the train generator
train_generator = train_datagen.flow_from_directory(
    train_data_path, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='binary'
)

#building the test generator
test_generator = test_datagen.flow_from_directory(
    test_data_path, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='binary'
)

print("\nData preprocessing completed successfully!\n")