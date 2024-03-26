import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
  import tf_keras as keras
else:
  keras = tf.keras
  
loaded_model = keras.models.load_model("dog_cat_classifier.keras", compile=False, custom_objects={'KerasLayer': hub.KerasLayer})

def process_image(image):
    # with keras
    img = keras.preprocessing.image.load_img(image, target_size=(224, 224))
    img = keras.preprocessing.image.img_to_array(img)
    # expected shape=(None, 224, 224, 3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    
    
    return img

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a page:', 
                           ['Prediction', 'Code', 'About'])

if options == 'Prediction': # Prediction page
    st.title('Dog vs Cat Classification with Transfer Learning')


    # User inputs: image
    image = st.file_uploader('Upload an image:', type=['jpg', 'jpeg', 'png'])
    if image is not None:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict'):
        with st.spinner('Model working....'):
            img_array = process_image(image)
            prediction = loaded_model.predict(img_array).argmax()
            if prediction == 0:
                st.write('Prediction: Cat')
            else:
                st.write('Prediction: Dog')
            
        
    
            
elif options == 'Code':
    st.header('Code')
    # Add a button to download the Jupyter notebook (.ipynb) file
    notebook_path = 'dog_cat_model.ipynb'
    with open(notebook_path, "rb") as file:
        btn = st.download_button(
            label="Download Jupyter Notebook",
            data=file,
            file_name="dog_cat_model.ipynb",
            mime="application/x-ipynb+json"
        )
    st.write('You can download the Jupyter notebook to view the code and the model building process.')
    st.write('--'*50)

    st.header('GitHub Repository')
    st.write('You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Dog-vs-Cat-Classification)')
    st.write('--'*50)
    
elif options == 'About':
    st.title('About')
    st.write('This web app is a simple image classification app that uses a pre-trained model to classify images of dogs and cats.')
    st.write('The model is trained using the MobileNet V2 architecture with ImageNet pre-trained weights.')
    st.write('This is a SavedModel in TensorFlow 2 format. Using it requires TensorFlow 2 (or 1.15) and TensorFlow Hub 0.5.0 or newer.')

    st.write('--'*50)
    st.write('The web app is open-source. You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Dog-vs-Cat-Classification)')
    st.write('--'*50)

    st.header('Contact')
    st.write('You can contact me for any queries or feedback:')
    st.write('Email: gokulnpc@gmail.com')
    st.write('LinkedIn: [Gokuleshwaran Narayanan](https://www.linkedin.com/in/gokulnpc/)')
    st.write('--'*50)
