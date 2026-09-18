import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")

with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Iris Flower Classifier")
st.write("Çiçeğin ölçümlerini gir, model türünü tahmin etsin.")

presets = {
    "Manuel gir": None,
    "Örnek: setosa": (5.1, 3.5, 1.4, 0.2),
    "Örnek: versicolor": (6.0, 2.9, 4.3, 1.3),
    "Örnek: virginica": (6.5, 3.0, 5.8, 2.2),
}
choice = st.selectbox("Hazır örnek seç (opsiyonel)", list(presets.keys()))
defaults = presets[choice] if presets[choice] else (5.8, 3.0, 3.8, 1.2)

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.3, 7.9, defaults[0], 0.1)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.4, defaults[1], 0.1)
with col2:
    petal_length = st.slider("Petal length (cm)", 1.0, 6.9, defaults[2], 0.1)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, defaults[3], 0.1)

if st.button("Tahmin et"):
    x = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(x)[0]
    st.success(f"Tahmin edilen tür: **{prediction}**")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        st.write("Sınıf olasılıkları:")
        for cls, p in zip(model.classes_, proba):
            st.write(f"- {cls}: {p:.2%}")
