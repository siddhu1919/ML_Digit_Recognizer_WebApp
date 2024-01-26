import gradio as gr
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model.keras')

def predict(img):
    if img is not None:
        img = img.reshape((1,28,28,1)).astype('float32') / 255

        prediction = model.predict(img)

        return {str(i) : float(prediction[0][i]) for i in range(10) }
    else:
        return ''
    
iface = gr.Interface(fn = predict , 
                    inputs = gr.Image(shape=(28,28),image_mode='L',invert_colors='True',source='canvas') ,
                    outputs = gr.Label(num_top_classes=3),
                    live=True ).launch()
