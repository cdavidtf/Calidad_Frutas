import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# Configuración de la pestaña del navegador
st.set_page_config(page_title="Detector de Frutas IA", page_icon="🍎", layout="centered")

st.title("🍎 Sistema Inteligente de Inspección de Frutas")
st.write("Sube una foto o captura desde la cámara para analizar el estado de la fruta en tiempo real.")
st.subheader("Proyecto de Inteligencia Artificial")
st.caption("Desarrollado por: Carlos David Torres Flores, Aaron Emiliano Hurtado Lopez, Edgar Rojas Reyes, Rafael Sebastian Jauregui")
st.sidebar.title("📌 Acerca del Proyecto")
st.sidebar.write("Este sistema utiliza una red neuronal profunda basada en **MobileNetV2** mediante *Transfer Learning* y *Fine-Tuning* para clasificar 18 tipos de frutas y verduras en estados óptimos o de descomposición.")
st.sidebar.info("🎯 Precisión del modelo en pruebas: ~90%")

# 1. Cargar el modelo (con caché para que cargue instantáneo la segunda vez)
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Descargando el modelo de Inteligencia Artificial desde Google Drive... Esto toma unos segundos.'):
            # Enlace limpio directo para descargas de Drive
            url = 'https://drive.google.com/uc?id=1u2S_yVDigDwl9bbEoEKTr6VVymsJs1zn'
            gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

# Activamos el cerebro de la IA
model = load_my_model()


# 2. Lista de tus 18 clases exactas (¡Asegúrate de que coincida con tus carpetas!)
nombres_clases = [
    'Manzana Fresca 🍎', 'Plátano Fresco 🍌', 'Melón Amargo Fresco 🥒', 'Pimiento Fresco 🫑',
    'Pepino Fresco 🥒', 'Okra Fresca 🌱', 'Naranja Fresca 🍊', 'Papa Fresca 🥔', 'Tomate Fresco 🍅',
    'Manzana Podrida 🍂', 'Plátano Podrido 🍂', 'Melón Amargo Podrido 🍂', 'Pimiento Podrido 🍂',
    'Pepino Podrido 🍂', 'Okra Podrida 🍂', 'Naranja Podrida 🍂', 'Papa Podrida 🍂', 'Tomate Podrido 🍂'
]

if model is None:
    st.error(
        "❌ No se encontró el archivo 'modelo_frutas_perfecto.keras' en esta carpeta. Asegúrate de que el nombre sea idéntico.")
else:
    st.sidebar.success("✅ Modelo de IA cargado correctamente")

    # 3. Selector para que el usuario elija si quiere subir archivo o usar Cámara
    opcion = st.radio("Selecciona el método de entrada:",
                      ("Subir una foto desde el dispositivo", "Usar la Cámara en vivo"))

    archivo_imagen = None
    if opcion == "Subir una foto desde el dispositivo":
        archivo_imagen = st.file_uploader("Elige una imagen de tu galería...", type=["jpg", "jpeg", "png"])
    else:
        archivo_imagen = st.camera_input("Apunta a la fruta con tu cámara")

    # 4. Procesar la imagen si existe
    if archivo_imagen is not None:
        imagen = Image.open(archivo_imagen)

        # Si la imagen viene de la cámara, a veces necesita convertirse a RGB
        if imagen.mode != "RGB":
            imagen = imagen.convert("RGB")

        st.image(imagen, caption='📷 Imagen a evaluar', use_container_width=True)

        # Preprocesamiento idéntico al de Google Colab (224x224 y normalizado)
        img_resized = imagen.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predicción de la Inteligencia Artificial
        with st.spinner('🕵️‍♂️ Analizando pixeles de la fruta...'):
            predicciones = model.predict(img_array)
            clase_idx = np.argmax(predicciones[0])
            confianza = predicciones[0][clase_idx] * 100
            resultado = nombres_clases[clase_idx]

        # 5. Desplegar los resultados de forma visual e impactante
        st.markdown("---")
        st.subheader("📊 Diagnóstico del Sistema:")

        # Si dice "rotten" (podrido) lo pintamos en rojo, si es "fresh" en verde
        if "podrida" in resultado.lower() or "podrido" in resultado.lower():
            st.error(f"## **{resultado.upper()}**")
        else:
            st.success(f"## **{resultado.upper()}**")

        st.metric(label="Porcentaje de Certeza", value=f"{confianza:.2f}%")
