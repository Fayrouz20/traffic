import os
os.environ["PAFY_BACKEND"] = "internal"

import streamlit as st
import torch
from train import Model  # Assurez-vous que le fichier train.py est dans le même répertoire
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon

# Fonction pour charger le modèle
def load_model(model_path, input_dims, fc1_dims, fc2_dims, n_actions):
    model = Model(lr=0.001, input_dims=input_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_actions=n_actions)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Fonction principale de Streamlit
def main():
    st.title("Déploiement du Modèle de Simulation de Trafic")
    
    st.sidebar.header("Paramètres du modèle")
    model_path = st.sidebar.text_input("Chemin du modèle", "model.pth")
    input_dims = st.sidebar.number_input("Dimensions d'entrée", 128)
    fc1_dims = st.sidebar.number_input("Dimensions de la première couche cachée", 256)
    fc2_dims = st.sidebar.number_input("Dimensions de la deuxième couche cachée", 256)
    n_actions = st.sidebar.number_input("Nombre d'actions", 4)

    model = None
    if st.sidebar.button("Charger le modèle"):
        try:
            model = load_model(model_path, input_dims, fc1_dims, fc2_dims, n_actions)
            st.success("Modèle chargé avec succès")
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle : {e}")

    st.header("Visualisation des résultats")
    if model:
        fig, ax = plt.subplots()
        # Ici, vous pouvez ajouter du code pour visualiser les prédictions du modèle
        # Ex: ax.plot(data, predictions)
        st.pyplot(fig)
    else:
        st.warning("Veuillez charger un modèle pour afficher les résultats")

if __name__ == "__main__":
    main()
