import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn

st.set_page_config(
    page_title="Predicción de Aprobación para Estudiantes",
    layout="centered"
)

st.title("Predicción de Aprobación para Estudiantes")
st.write(
    "Esta aplicación carga un Pipeline registrado en MLflow."
)

# Conexión al servidor MLflow
mlflow.set_tracking_uri("http://127.0.0.1:9090")

# versión del modelo:
MODEL_URI = "models:/prediccionAprobacion/10"

@st.cache_resource
def cargar_modelo():
    return mlflow.sklearn.load_model(MODEL_URI)

model = cargar_modelo()

st.sidebar.header("Configuración")
st.sidebar.write(f"Modelo cargado: `{MODEL_URI}`")

st.subheader("Datos del estudiante")

col1, col2 = st.columns(2)

with col1:
    carrera = st.selectbox("Carrera", ["Computacion", "Derecho", "Economia", "Medicina", "Arquitectura", "Industrial"])
    modalidad = st.selectbox("Modalidad", ["Presencial", "Virtual", "Hibrida"])
    beca = st.selectbox("¿Tiene beca?", ["Si", "No"])

with col2:
    edad = st.number_input("Edad", min_value=18, max_value=30, value=18)
    promedio = st.number_input("Promedio", value=7.0, min_value=4.0, max_value=10.0, step=0.1)
    asistencias = st.number_input("Asistencia", value=60, min_value=0, max_value=100, step=1)

datos = pd.DataFrame([{
    "carrera": carrera,
    "modalidad": modalidad,
    "beca": beca,
    "edad": edad,
    "promedio": promedio,
    "asistencias": asistencias
}])

st.subheader("Datos enviados al modelo")
st.dataframe(datos)

if st.button("Predecir"):
    prediccion = model.predict(datos)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(datos)[0]
        prob_no = proba[0]
        prob_si = proba[1]
    else:
        prob_no = None
        prob_si = None

    if prediccion == 1:
        st.success("Predicción: el estudiante SÍ podría aprobar.")
    else:
        st.warning("Predicción: el estudiante NO aprobaría.")

    if prob_si is not None:
        st.write(f"Probabilidad de NO aprobar: {prob_no:.4f}")
        st.write(f"Probabilidad de SÍ aprobar: {prob_si:.4f}")

st.caption(
    "Práctica 2. Elaborada por Edison Moisés Luzardo Delgado"
)
