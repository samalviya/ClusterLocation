import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm

import streamlit as st
from streamlit_folium import folium_static
import folium

import random
from sklearn.cluster import KMeans

st.sidebar.title("Input Parameters")

clusterNumber = st.sidebar.number_input('Number of clusters to be created', 1, 100, step=1)
clusterNumber = st.sidebar.slider('Adjust no. of Cluster', 0, clusterNumber * 2, clusterNumber)
uploaded_file = st.sidebar.file_uploader("Upload CSV file with GPS Coordinates")
pointer = st.sidebar.checkbox('Show centre pin of each cluster')

csv = pd.read_csv('Sample.csv')

st.download_button(
    label="Download Sample CSV",
    data=csv.to_csv(),
    file_name='Sample.csv',
    mime='text/csv',
)

download = pd.DataFrame()

if clusterNumber < 2:
    st.error("Please enter the number of clusters")
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df[['Latitude', 'Longitude']] = df['GPS'].str.split(',', expand=True).astype(float)
        df = df[['ID', 'Latitude', 'Longitude']]
    except:
        st.error("Sheet is in wrong format")

    try:
        kmeans = KMeans(n_clusters=int(clusterNumber), init='k-means++')
        kmeans.fit(df[df.columns[1:3]])
        df['cluster_label'] = kmeans.fit_predict(df[df.columns[1:3]])
        download = df.copy()
        download['cluster_label'] = download['cluster_label'] + 1
        centers = kmeans.cluster_centers_
        labels = kmeans.predict(download[download.columns[1:3]])
        download['GPS'] = download['Latitude'].astype(str) + ',' + download['Longitude'].astype(str)
        download = download[['ID', 'GPS', 'cluster_label']]

        st.success("Cluster map is created, you can download or see it")

        st.download_button(
            label="Download Output",
            data=download.to_csv().encode('utf-8'),
            file_name='Clustered_GPS.csv',
            mime='text/csv',
        )

        center_lat = df['Latitude'].mean()
        center_lon = df['Longitude'].mean()

        min_lat, max_lat = df['Latitude'].min(), df['Latitude'].max()
        min_lon, max_lon = df['Longitude'].min(), df['Longitude'].max()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=10, zoom_control=False)

        df['cluster_label'] = df['cluster_label'].astype(int)

        num_clusters = len(np.unique(df['cluster_label']))
        cluster_colors = [colors.rgb2hex(cm.viridis(i / num_clusters)) for i in range(num_clusters)]

        for idx, row in df.iterrows():
            cluster_color = cluster_colors[int(row['cluster_label'])]
            folium.CircleMarker(location=[row['Latitude'], row['Longitude']], radius=2, color=cluster_color).add_to(m)

        folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri').add_to(m)
        folium.TileLayer('cartodbpositron').add_to(m)
        
        if pointer:
            colour = ['lightgreen', 'green', 'darkpurple', 'gray', 'lightred', 'darkred', 'black', 'purple', 'lightblue', 'blue', 'orange', 'cadetblue', 'red', 'lightgray', 'pink', 'darkgreen', 'darkblue', 'beige']
            for center, color in zip(centers, cluster_colors):
                folium.Marker(location=[center[0], center[1]], icon=folium.Icon(color=random.choice(colour))).add_to(m)
        
        folium.LayerControl().add_to(m)
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

        folium_static(m)
    except Exception as e:
        st.warning(f"Something went wrong! Contact Support. Error: {e}")
else:
    st.warning("Please Upload CSV")

if not download.empty:
    st.write(download)
