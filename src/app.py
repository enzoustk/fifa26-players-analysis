import ast
import pandas as pd
import streamlit as st
from streamlit_features.gmm_code import *
from streamlit_features.k_means_code import *
from streamlit_features.data import data_cleaning, eda

df = pd.read_pickle('src/data/df_cleaned.pkl')


st.title('Soccer Player Analysis')

tabs = st.tabs([
    'About the Project',
    'Data Cleaning Feature Engineering',
    'Exploratioral Data Analysis',
    'K-Means Model',
    'GMM Model'
])

with tabs[0]: #Resumo
    st.markdown("""
    Welcome to the **Soccer Player Analysis** platform. This project was developed as the final capstone for the **Unsupervised Machine Learning** course (IMD3003) at the Federal University of Rio Grande do Norte (UFRN), under the supervision of **Professor Silvan Ferreira**.

    Our goal is to move beyond standard player ratings and use Data Science to uncover hidden patterns in athlete performance, simulating a scouting environment for the upcoming **FIFA 26** cycle.

    ---

    ### Objetive:

    In a dataset containing thousands of athletes, how can we objectively distinguish a "Clinical Finisher" from a "Box-to-Box Midfielder" based solely on data? By applying **Clustering algorithms**, we group players not by their pre-assigned positions, but by their **actual statistical DNA**.

    ### Technical Core

    To ensure high-precision groupings, this application leverages a robust Machine Learning pipeline:

    * **Gaussian Mixture Models (GMM):** Our primary model, chosen for its flexibility in capturing clusters of various shapes and providing soft-clustering probabilities.
    * **K-Means:** Used as a baseline to validate cluster consistency and define rigid centroids.
    * **Dimensionality Reduction (PCA & UMAP):** We compress dozens of technical attributes into 2D and 3D projections, allowing us to visualize the "spatial distance" between different playing styles.

    ---

    ### Key Features

    1. **Player Segmentation:** Explore how the dataset is divided into meaningful profiles such as *Elite Veterans*, *High-Potential Prospects*, and *Tactical Specialists*.
    2. **Attribute Distribution:** Analyze how traits like acceleration, vision, and physical strength vary across different clusters.
    3. **Interactive Visualizations:** Navigate through UMAP projections to find "hidden gems"—players who statistically resemble world-class stars.

    ### Methodology

    The project follows a rigorous Data Science workflow:

    1. **Preprocessing:** Data cleaning, handling missing values, and feature scaling to ensure no single attribute dominates the model.
    2. **Feature Selection:** Identifying the most relevant metrics for on-pitch performance.
    3. **Cluster Interpretation:** Translating mathematical labels into actionable football insights.

    ---

    > **Developed by:** [Enzo Araújo]
    > **Source Code:** [GitHub Repository](https://github.com/enzoustk/fifa26-players-analysis)

    """
)


with tabs[1]: # Dataset
    data_cleaning(df=df.copy())

with tabs[2]: # Exploratiorial Data Analysis
    eda(df=df.copy())

with tabs[3]: # K-Means
    kmeans_df = plot_kmeans()
    st.subheader('Results')
    
    kmeans_tabs = st.tabs([
        'Top 10 Players By Cluster',
        'Distribution of Ratings by Atribute',
        'Technical Report',
        'Code'
        ]
    )
    with kmeans_tabs[0]:
        cluster_df(df=kmeans_df)
    
    with kmeans_tabs[1]:
        plot_kde(df_viz=kmeans_df)

    with kmeans_tabs[2]:
        st.markdown(tech_report)

    with kmeans_tabs[3]:
        st.write("Step 1: Run Principal Component Analysis using 3 Principal Components to be able to make 3d plots")
        st.code(pca_code, language='python')
        st.write("Step 2: Run KMeans with K==4 (Elbow Method)")
        st.code(kmeans_code, language='python')

with tabs[4]: # Gaussian Models Mixture
    umap_data = plot_umap()
    gmm_df = plot_gmm()
    
    gmm_tabs = st.tabs([
        'Cluster Report',
        'Cluster Feature Matrix',
        'Code'
    ])

    with gmm_tabs[0]:
        st.markdown(gmm_report)
        st.divider()
        st.subheader('Top 5 Players by Cluster')
        top_gmm_players(df=gmm_df)

    with gmm_tabs[1]:
        cluster_feature_matrix(df=gmm_df)

    
    with gmm_tabs[2]:
        st.code(umap_code, language='python')
        st.code(gmm_code, language='python')

st.divider()

cols = st.columns([2,8])
with cols[0]: st.write('About the dataset:')
with cols[1]:
    st.page_link(
        label='EAFC26 Mens Player Data Analysis Modeling (Click to view on Kaggle)',
        page='https://www.kaggle.com/code/devraai/eafc26-mens-player-data-analysis-modeling'
    )
