import tensorflow as tf
from tensorflow.keras import layers

def build_conv_lstm_model(input_shape):
    """
    Builds a ConvLSTM model for spatio-temporal data processing using TensorFlow and Keras.
    
    This model is suitable for tasks where both spatial and temporal dependencies need to 
    be captured, such as video prediction or time-series forecasting for demand prediction.

    Parameters:
    - input_shape (tuple): The shape of the input data. Should be a 5D tensor representing 
      (timesteps, height, width, channels) where:
        - timesteps: Number of time steps (sequence length).
        - height: Height of the spatial grid
        - width: Width of the spatial grid 
        - channels: Number of input channels 

    Returns:
    - model (tensorflow.keras.Model): A compiled Keras model ready for training.
    """


    model = tf.keras.Sequential([


        layers.Input(shape=input_shape),  
        layers.ConvLSTM2D(
            64,                 
            kernel_size=(3, 3),  
            activation='relu',    
            padding='same', 
            return_sequences=True 
        ),
        layers.BatchNormalization(), 
        layers.ConvLSTM2D(
            32,                 
            kernel_size=(3, 3),    
            activation='relu',     
            padding='same'        
        ),
        layers.BatchNormalization(), 
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  
    ])


    model.compile(optimizer='adam', loss='mae')
    
    return model


