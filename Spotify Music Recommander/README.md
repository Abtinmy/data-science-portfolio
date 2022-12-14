# Overview
### Spotify Cluster-Based Music Recommander and Genre Classifier
Creating a music genre classifier, and Performing different clustering algorithms on Spotify music dataset on Kaggle. Using user-generated data by calling Spotify API and recommand a playlist generated by calculating the distance of user preferences and the clusters.
 
Tech: Python, Scikit-Learn, Pandas, Matplotlib, Seaborn

![](resources/spotify.png)

# Methodology
In this project, I created a terminal-based python application capable of reading a dataset as the input consisting of feature analyses of some music that a specific user like which is provided by Spotify’s API. The application uses a classifier trained using the large dataset consisting of available music on Spotify and their genre to predict genres of music in the input dataset. Then music will be divided into different clusters by a clustering algorithm which is trained with the main dataset. After that, five playlists are recommended to the user based on total number of input music in each cluster.

1. Download and preprocess Spotify music dataset which is provided on Kaggle.
2. Train a classifier that can determine genres of each music and extract importance of every feature.
3. Run the classifier on user’s input dataset.
4. Find and train the best clustering algorithm that can divide music in the dataset into well-separated clusters.
5. Run the trained cluster algorithm on input dataset.
6. Choose the top five clusters for each user and recommend related music in those clusters.

# Credits
- [Dataset on Kaggle](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify)
- [Dataset Documentation](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-several-audio-features)
- [Final Reprot](report.pdf)