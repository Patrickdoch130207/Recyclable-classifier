import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.preprocessing.image import img_to_array,load_img

@st.cache_data()
def load():
    model_path = "best_model.keras"
    model = load_model(model_path, compile = False)
    return model


# Chargement du model
model = load()




def predict(upload):
    if upload is not None:
        img = Image.open(upload)
        img = np.asarray(img)
        
        # Préparation de l'image
        img_resize = cv2.resize(img, (224, 224))
        img_resize = np.expand_dims(img_resize, axis=0)
      
        # Prédiction
        pred = model.predict(img_resize)
        
        rec = pred[0][0]
        return rec
    return None

# Interface utilisateur
st.title("Poubelle Intelligente")

upload = st.file_uploader("Charger l'image de votre objet", 
                          type=["jpg", "jpeg", "png"])

if upload is not None:
    # Afficher l'image uploadée
    st.image(upload, caption='Image chargée', use_container_width=True)
    
    # Faire la prédiction
    prediction = predict(upload)
    
    if prediction is not None:
        # Afficher le résultat
        st.write("Prédiction :", prediction)
        if prediction > 0.5:
            st.success("Cet objet est recyclable!")
        else:
            st.error("Cet objet n'est pas recyclable.")
