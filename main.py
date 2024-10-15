import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm

import streamlit as st
from streamlit_folium import folium_static
import folium

import random
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Set up Streamlit app
st.title("GPS Clustering Tool")
st.subheader("Upload your GPS data and visualize clusters on the map")
st.markdown("### Instructions:")
st.markdown("1. Upload a CSV file containing GPS coordinates in a column named 'GPS' (formatted as 'latitude,longitude').")
st.markdown("2. Choose the number of clusters or adjust other parameters for DBSCAN.")
st.markdown("3. Optionally, display the center of each cluster on the map.")

# Sidebar settings
with st.sidebar:
    st.header("Upload Data & Settings")
    uploaded_file = st.file_uploader("Upload CSV file with GPS Coordinates", type=["csv"])
    clustering_method = st.selectbox("Select Clustering Algorithm", ["K-Means", "DBSCAN", "Agglomerative Clustering"])

    # Additional parameters for DBSCAN
    if clustering_method == "DBSCAN":
        epsilon = st.number_input("Epsilon (max distance between points)", min_value=0.001, max_value=1.0, step=0.001, value=0.1)
        min_samples = st.slider("Minimum Samples per Cluster", 1, 20, 5)
    else:
        clusterNumber = st.slider('Number of clusters', 1, 100, step=1)

    pointer = st.checkbox('Show cluster center markers', help="Display the centroid of each cluster on the map.")
    show_dendrogram = st.checkbox('Show Dendrogram (Agglomerative Only)', value=False)
    st.markdown("---")
    st.markdown("### Download Options")

    # Sample CSV download
    sample_csv = pd.DataFrame({"ID": [1, 2, 3], "GPS": ["22.666838,77.641035", "22.665000,77.640000", "22.668000,77.642000"]})
    st.download_button(
        label="Download Sample CSV",
        data=sample_csv.to_csv(index=False),
        file_name='Sample.csv',
        mime='text/csv'
    )

# Initialize download DataFrame
download = pd.DataFrame()

# Main logic
if uploaded_file is not None:
    try:
        # Read uploaded file
        df = pd.read_csv(uploaded_file)
        if 'GPS' not in df.columns:
            st.error("The CSV must contain a 'GPS' column.")
            st.stop()

        # Split GPS column into Latitude and Longitude
        df[['Latitude', 'Longitude']] = df['GPS'].str.split(',', expand=True).astype(float)
        df = df[['ID', 'Latitude', 'Longitude']]

        # Choose clustering algorithm
        if clustering_method == "K-Means":
            if clusterNumber < 2:
                st.error("K-Means requires at least 2 clusters.")
                st.stop()
            model = KMeans(n_clusters=clusterNumber, init='k-means++')
            df['cluster_label'] = model.fit_predict(df[['Latitude', 'Longitude']])
            centers = model.cluster_centers_
        elif clustering_method == "DBSCAN":
            model = DBSCAN(eps=epsilon, min_samples=min_samples)
            df['cluster_label'] = model.fit_predict(df[['Latitude', 'Longitude']])
            centers = None
        else:  # Agglomerative Clustering
            model = AgglomerativeClustering(n_clusters=clusterNumber)
            df['cluster_label'] = model.fit_predict(df[['Latitude', 'Longitude']])
            centers = None

            # Optionally show dendrogram
            if show_dendrogram:
                st.subheader("Dendrogram")
                Z = linkage(df[['Latitude', 'Longitude']], 'ward')
                plt.figure(figsize=(10, 7))
                dendrogram(Z)
                st.pyplot(plt)

        # Mark noise points in DBSCAN
        if clustering_method == "DBSCAN":
            df['cluster_label'] = np.where(df['cluster_label'] == -1, 'Noise', df['cluster_label'])

        # Prepare download DataFrame
        download = df.copy()
        download['GPS'] = download['Latitude'].astype(str) + ',' + download['Longitude'].astype(str)
        download = download[['ID', 'GPS', 'cluster_label']]

        # Success message
        st.success("Cluster map is created, you can download or see it")

        # Download output button
        st.download_button(
            label="Download Output",
            data=download.to_csv(index=False).encode('utf-8'),
            file_name='Clustered_GPS.csv',
            mime='text/csv',
        )

        # Map creation
        center_lat = df['Latitude'].mean()
        center_lon = df['Longitude'].mean()
        min_lat, max_lat = df['Latitude'].min(), df['Latitude'].max()
        min_lon, max_lon = df['Longitude'].min(), df['Longitude'].max()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        # Color setup
        num_clusters = len(np.unique(df['cluster_label'])) if clustering_method != "DBSCAN" else len(df['cluster_label'].unique())
        cluster_colors = [colors.rgb2hex(cm.viridis(i / max(num_clusters - 1, 1))) for i in range(num_clusters)]

        # Plot clusters
        for idx, row in df.iterrows():
            cluster_color = 'gray' if row['cluster_label'] == 'Noise' else cluster_colors[int(row['cluster_label']) % len(cluster_colors)]
            folium.CircleMarker(location=[row['Latitude'], row['Longitude']], radius=2, color=cluster_color).add_to(m)

        # Add tile layers
        folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri').add_to(m)
        folium.TileLayer('cartodbpositron').add_to(m)

        # Optionally add center markers
        if pointer and centers is not None:
            colour = ['lightgreen', 'green', 'darkpurple', 'gray', 'lightred', 'darkred', 'black', 'purple', 'lightblue', 'blue', 'orange', 'cadetblue', 'red', 'lightgray', 'pink', 'darkgreen', 'darkblue', 'beige']
            for center in centers:
                folium.Marker(location=[center[0], center[1]], icon=folium.Icon(color=random.choice(colour))).add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Fit map to data bounds
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

        # Show map
        folium_static(m)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please upload a CSV file to proceed.")

# Display the download DataFrame if available
if not download.empty:
    st.write(download)
