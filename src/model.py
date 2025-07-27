from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization

def build_model(input_shape=(224, 224, 3)):
    """
    Builds the VGG16-based CNN model with a custom classifier head.

    Args:
        input_shape (tuple): The input shape of the images.

    Returns:
        A Keras Model instance.
    """
    # Load VGG16 base model, pre-trained on ImageNet, without the top classifier layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # --- Custom Classifier Head ---
    # Get the output of the base model
    x = base_model.output
    
    # Add custom layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x) # Stabilizes the network
    x = Dropout(0.5)(x) # Regularization
    
    # Final output layer for binary classification
    predictions = Dense(1, activation='sigmoid')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    print("✅ Model architecture built successfully.")
    
    return model