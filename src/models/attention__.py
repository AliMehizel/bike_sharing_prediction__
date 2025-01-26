import tensorflow as tf
from tensorflow.keras import layers

def attention_lstm_model(input_shape):
    """
    Build a model with an attention layer followed by an LSTM layer.
    
    Parameters:
    input_shape: tuple
        The shape of the input data, should be (timesteps, features).
        
    Returns:
    model: tensorflow.keras.Model
        The constructed model with Attention and LSTM layers.
    """
    inputs = layers.Input(shape=input_shape)


    attention = layers.Attention()([inputs, inputs])  


    lstm = layers.LSTM(64, return_sequences=False)(attention)
    

    output = layers.Dense(1)(lstm)
    

    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    model.compile(optimizer='adam', loss='mae')
    
    return model
