
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

'''
https://medium.com/swlh/hands-on-the-cifar-10-dataset-with-transfer-learning-2e768fd6c318
'''

def getModelNames():
    return ["VGG16", "VGG19", "ResNet50V2", "ResNet152V2", "Xception", "InceptionResNetV2", "DenseNet121", "DenseNet169"] #, "EfficientNetB7"]

def resnet50(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS):
    
    base_model =  tf.keras.applications.ResNet50(
        weights = 'imagenet', 
        include_top = False, 
        input_shape = (HEIGHT, WIDTH, CHANNELS))

    for layer in base_model.layers:
        layer.trainable = False
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(1000, activation='relu')(x)
    predictions = layers.Dense(NUM_OUTPUTS, activation = 'softmax')(x)
    model = keras.Model(inputs = base_model.input, outputs = predictions)
    return model

def model(model_name, HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS):

    MODELS = {
        "VGG16": tf.keras.applications.VGG16,
        "VGG19": tf.keras.applications.VGG19,
        "ResNet50V2": tf.keras.applications.ResNet50V2,
        "ResNet152V2": tf.keras.applications.ResNet152V2,
        "Xception": tf.keras.applications.Xception, 
        "InceptionResNetV2": tf.keras.applications.InceptionResNetV2,
        "DenseNet121": tf.keras.applications.DenseNet121,
        "DenseNet169": tf.keras.applications.DenseNet169,
        #"EfficientNetB7": tf.keras.applications.EfficientNetB7
    }
    Network = MODELS[model_name]
    base_model = Network(weights = 'imagenet', 
                        include_top = False, 
                        input_shape = (HEIGHT, WIDTH, CHANNELS),
                        pooling='max')

    for layer in base_model.layers:
        layer.trainable = False
    x = layers.Flatten()(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(NUM_OUTPUTS, activation='softmax')(x)
    model = keras.Model(inputs = base_model.input, outputs = x)
    model.base_model = base_model # Save ref to base_model 

    print("Number of layers in the base model: ", len(base_model.layers))
    return model

def fine_tune_model(model_name, model):
    
    MODELS = {
        "VGG16": 6,
        "VGG19": 6,
        "ResNet50V2": 25,
        "ResNet152V2": 50,
        "Xception":  25, 
        "InceptionResNetV2": 50,
        "DenseNet121":  40,
        "DenseNet169": 60,
        #"EfficientNetB7":  6,
    }

    base_model = model.base_model
    base_model.trainable = True

    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine-tune from this layer onwards
    fine_tune_at =  len(base_model.layers) - MODELS[model_name]
    print("FineTune at ", fine_tune_at)

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    model.compile(optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9),   
                 loss='sparse_categorical_crossentropy', metrics=['acc']) 
    return model 