# Project Description
This file describes the steps we have taken in data preparation and model development in this music recommendation system project. The submission folder includes all finalized notebook files, while other folders contain our prior explorations and intermediate files. 


# Data Collection, Preprocessing, Feature Engineering, Cleaning, and EDA Steps

This part introduces the steps we have taken to prepare data for ML models, the steps are as follows:

## 1. Data Collection
Collect user and song data from Last.fm API (Step1_DataCollectionPreprocessing.ipynb -> Constructing the main user-song dataset).

[Set up API calls] => [Get a minimum of 10000 users and 100000 unique songs] => [For each user, obtain their top 50 songs, various demographics information, usage information on last.fm platform] => [lastfm_user_raw.csv] => [data cleaning to drop invalid rows and users with no song information] => [lastfm_user_clean.csv]

## 1.1. Unique Songs (feature extraction and preprocessing)
Collect meta information of our users' unique top songs using Last.fm API. ((Step1_DataCollectionPreprocessing.ipynb -> Constructing Unique Songs Dataset)

[Data collection from API] => [song_details_raw.csv] => [Cleaning song details data, keep song information including track_name, artist_name, duration, listeners, playcount, and toptags] => [song_details_clean.csv]

### 1.2 Lyrics Retrieval and Feature Extraction
Fetch song lyrics using Genius API using parallel processing. (Step1_DataCollectionPreprocessing.ipynb -> Retrieving Lyrics, Step1A_Lyrics.ipynb)

[Preprocess the lyrics and drop non-English songs and invalid lyrics] => [song_details_lyrics_clean.csv] => Feature engineering for songs, NLP on lyrics to extract features like primary emotions, profanity, polarity, subjectivity. => [songs_details_lyrics_FE_clean.csv]


### 1.3 Audio Features Retrieval and Feature Extraction
Fetch 30-second audio of songs using deezer API, and extract features from these audio using Librosa API. (step1B_AudioFeatures.ipynb) => Extract audio features such as mfcc, chroma, rms, spectral_centroid, zcr (detailed feature explanation below)


### 1.4 Combining All Song Features
Combine all features created above. (Step1_DataCollectionPreprocessing.ipynb -> Putting all song info together) [unique_songs_features_complete.csv]
(Step1_DataCollectionPreprocessing.ipynb -> Putting all song info together) 
users and their top songs with features: [user_songs_filtered.csv]


**Feature Descriptions**:
The dataset contains audio features and lyrics features of songs and information (artist, title, number of listeners and playcount) we collated from various sources.

- listeners: number of listeners recorded on last.fm
- total_playcount: total number of playcount recorded on last.fm
- top_tags: top tags for a song, they can be user-generated on last.fm
- profanity_density: profanity level from feature engineering of the lyrics
- emotion1/2: top emotions of the song from feature engineering of the lyrics 
- Zero Crossing Rate: the rate at which the sound signal changes from positive to negative and vice versa. This feature is usually used for speech recognition and music information retrieval. Music genre with high percussive sound like rock or metal usually have high Zero Crossing Rate than other genres.
- Tempo BMP (beats per minute): Tempo is the number of beat per one minute.
- Spectral Centroid: This variable represents brightness of a sound by calculating the center of sound spectrum (where the sound signal is at its peak). We can also plot it into a wave form.
- Mel-Frequency Cepstral Coefficients: The Mel frequency Cepstral coefficients (MFCCs) of a signal are a small set of features that describes the overall shape of a spectral envelope. It imitates characteristics of human voice.
- Chroma Frequencies: Chroma feature represents the tone of music or sound by projecting its sound spectrum into a space that represents musical octave. This feature is usually used in chord recognition task.


### Sequential Data for LSTM model
(Step4D_LSTM.ipynb)
We extract users' recent tracks from lastfm. These pertain to User data, where we want to get their recent tracks from up to one month ago. We limit it to 100 tracks per user. This is run on unique users from the user-song dataset. 

Timestamp Conversion: The 'Timestamp' column, is converted into a datetime format. This allows for extracting more granular time information such as hours.

Time of Day: 'Time_of_Day' feature by extracting the hour from the timestamp. This is done to capture patterns in listening behavior based on the time of day.

Combining Artist and Track: The 'Artist' and 'Track Name' columns are combined into a single 'Artist_Track' column. This creates a unique identifier for each artist-track combination 

One-Hot Encoding Time of Day

Label Encoding Artist-Track: The 'Artist_Track' combinations are label-encoded, assigning a unique integer to each unique artist-track string. This is necessary for the model to process textual/categorical data.

Concatenate the original dataframe with the one-hot encoded 'Time_of_Day' features, expanding the feature set to include this time information explicitly.


## 3. EDA
(Step2_EDA.ipynb)

### 3.1 EDA for Users
Descriptive statistics of user features, as well as visualization of feature distribution and construction of categorical feature from numerical features, one-hot encoding, etc. 

### 3.2 EDA for Songs
Descriptive statistics of song features, visualization of feature distribution and construction of categorical feature from numerical features, one-hot encoding, log transformation, etc. 
Construct features based on domain knowledge of audio features mfcc and chroma: mean and variance of list of 20 and 12 numerical values respectively. 

### 3.3 Feature Engineering and PCA
Features constructed for users: log_total_track_count, log_total_artist_count, activeness (one-hot-encoding)
=> [users_feature_eng.csv]

Features constructed for songs:
log_listeners, log_total_playcount, profanity (low, medium, high), duration(short medium, long), emotion (positive, negative, joy, disgust, anger, fear, sadness, surprise, trust), scaled mfcc values. + 
Create features with domain understanding of audio features: mfcc_mean, mfcc_var, chroma_mean, chroma_var

Principal component analysis:
Obtain 6 principal components which explain about 82.17% of total variance in data. 
=> [songs_feature_eng_pca.csv]


# Model Development


## Cosine Similarity
(Step3_CosineSimilarity.ipynb)
We explore content-based filtering method. We use the 6 different Principal Components to find the top 5 most similar tracks using cosine similarity, denoted by 'angular' using the Annoy package. This method was run on the entire dataset and the output of this was then utlized in GCN models.

## KMeans
(Step3A_KMeans.ipynb)
We explore content-based filtering method. Songs are described with a set of lyrics and audio features, or the principal components generated during feature engineering. We assume users prefer to listen to songs that are similar to what they have listened to before. Given a song, we create a function to recommend the top 30 most similar songs from the same cluster.

## KNN
(Step4_KNN.ipynb)
We are implementing a collaborative filtering using a SVD approach in the `surprise` package. The surprise package also takes care of the train test split. The collaborative filter transforms user-item nteractions into latent space and reconstructs the user-item matrix to predict missing `relative_playcount` values based on the patterns it has learnt. The predicted rating is the dot product between the `Username` and `Track_Artist` vectors in latent space.

We used the `Username`, `track_name` and `artist_name` and `playcount` from the `user_songs_filtered.csv` dataset. The evaluation works by comparing the predicted songs using the SVD approach to the user's actual top 30 songs in the user's playlist. We are predicting missing values in the pivot table (Matrix) being created, which are songs that the user has not listened to.


## Implicit
(Step4A_Implicit.ipynb)
Recommendations are based on the users' behavior (e.g., playcounts of songs) rather than explicit ratings. The code is using a collaborative filtering approach with the Alternating Least Squares (ALS) algorithm. We recommend both seen and unseen songs (Unseen means between training and test set). We use `user_songs_filtered.csv` and AlternatingLeastSquares model from the implicit library, which primarily focuses on matrix factorization techniques that work directly with user-item interaction data. It doesn't natively support incorporating user or item features directly into the model during the matrix factorization process in the way that models like LightFM do, which are designed as hybrid recommendation models capable of utilizing both interaction data and metadata (e.g., genre, user demographics).

For model evaluation, we have the following metrics: 

Precision at K: This measures the proportion of recommended items in the top-K set that are actually relevant. "relevant" means that the item appears in the user's interaction data in the test set.

AUC: Model's ability to rank a randomly chosen positive item (an item the user has interacted with) higher than a randomly chosen negative item (an item the user has not interacted with).

NDCG: The Normalized Discounted Cumulative Gain accounts for the position of the relevant items in the recommendation list. It places higher importance on relevant items being positioned higher in the list.

MAP: The Mean Average Precision at K averages the precision at each rank for the relevant items and considers the order in which the relevant items appear.

If a song was played by a user, it is relevant to them. Songs from the test set that the user has played but were not in the training set are used as the ground truth for relevance.


## LightFM
(Step4B_LightFM.ipynb)
This model can only be run locally as there will be RAM limit error on Colab.
LightFM is a Python implementation of a Factorization Machine recommendation algorithm for both implicit and explicit feedbacks. It is a Factorization Machine model which represents users and items as linear combinations of their content features’ latent factors. The model learns embeddings or latent representations of the users and items in such a way that it encodes user preferences over items. These representations produce scores for every item for a given user; items scored highly are more likely to be interesting to the user.

The user and item embeddings are estimated for every feature, and these features are then added together to be the final representations for users and items. For example, for user i, the model retrieves the i-th row of the feature matrix to find the features with non-zero weights. The embeddings for these features will then be added together to become the user representation e.g. if user 10 has weight 1 in the 5th column of the user feature matrix, and weight 3 in the 20th column, the user 10’s representation is the sum of embedding for the 5th and the 20th features multiplying their corresponding weights. The representation for each items is computed in the same approach.

The detailed model development is described in Step4B_LightFM.ipynb.

Reference: https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/lightfm_deep_dive.ipynb


### LightFM Model Parameters Explanation
- `SEED = 42`: Sets the seed for pseudorandom number generation to ensure reproducibility of the model training and data splitting.
- `K = 30`: Defines the number of top recommendations to be evaluated (not used in training but often used in evaluation metrics like precision@k).
- `TEST_PERCENTAGE = 0.2`: Specifies that 20% of the data should be set aside as a test dataset to evaluate the model's performance.
- `LEARNING_RATE = 0.05`: Determines the step size at each iteration to minimize the loss function, controlling how quickly the model learns.
- `NO_COMPONENTS = 20`: Number of latent factors to use in the model, representing the dimensions in which users and items are characterized.
- `NO_EPOCHS = 50`: Indicates the number of complete passes through the training dataset the model should make during training.
- `NO_THREADS = 16`: Sets the number of parallel threads to use during model fitting, enhancing computational efficiency on multicore systems.
- `ITEM_ALPHA = 1e-6` and `USER_ALPHA = 1e-6`: Regularization parameters for item and user features to prevent overfitting by penalizing large weights in the model.

### Model Configuration
- The model is configured to use the WARP loss function, which optimizes the order of items in recommendations, focusing on ranking higher the items that should be recommended.

### Model Training
- The model is trained on the interaction data, incorporating user and item features to enhance the recommendations' relevance and accuracy.

### Model Serialization
- Post training, the model is serialized and saved to Google Drive, enabling the persistence of the trained model for future use without the need to retrain.


## NCF
(Step4C_NCF.ipynb)
This model can only be run locally as there will be RAM limit error on Colab.
Neural Collaborative Filtering (NCF) is an algorithm based on deep neural networks to tackle collaborative filtering on the basis of implicit feedback. Since we are using neural networks to find relation between users and items, we can easily scale the solution to large datasets. Thus making this method better than item based collaborative filtering.

NCF works by first representing users and items as vectors in a latent space. These vectors are then used to calculate a score for each user-item pair. The score is then used to predict whether the user will interact with the item. NCF is useful because it can learn non-linear relationships between users and items. This makes it a more powerful model than traditional matrix factorization methods.

Reference: [https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/ncf_deep_dive.ipynb]

We use stratified split, which is similar to random sampling, but the splits are stratified, for example if the datasets are split by user, the splitting approach will attempt to maintain the same ratio of items used in both training and test splits. The converse is true if splitting by item.

### NCF parameters:

`n_users`, number of users. We are one hot encoding our user data. Therefore the input size of the model will be number of users.

`n_items`, number of items. Same logic as `n_users`.

`batch_size`, number of examples you want the model to process at a time. Higher value will consume more memory.

`learning_rate`, this can be thought of as how much you want the model to change after one iteration. Large value will lead to unstability and very small values will take more time to converge.

`n_factors`, which controls the dimension of the latent space. Usually, the quality of the training set predictions grows with as n_factors gets higher.

`layer_sizes`, sizes of input layer (and hidden layers) of MLP, input type is list. We have set it to [64,32,16,8,4] as from training and testing, higher values gave better results.

`n_epochs`, which defines the number of iteration of the SGD procedure. Note that both parameter also affect the training time.

`model_type`, we can train single "MLP", "GMF" or combined model "NCF" by changing the type of model.

### Evaluating NCF
For a general evaluation, we remove songs that are already users' top songs in the top k recommendations. To compute ranking metrics, we need predictions on all user, item pairs. We do not want to recommend the same item again to the user.

As suggested by the reference paper, we also constructed dataset for "leave-one-out" evaluation: For each item in test data, we randomly samples 100 items that are not interacted by the user, ranking the test item among the 101 items (1 positive item and 100 negative items). The performance of a ranked list is judged by Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG). Finally, we average the values of those ranked lists to obtain the overall HR and NDCG on test data.

We truncated the ranked list at 10 for both metrics. As such, the HR intuitively measures whether the test item is present on the top-10 list, and the NDCG accounts for the position of the hit by assigning higher scores to hits at top ranks.

### NCF with Pre-trained MLP and GMF
To get better performance of NeuMF, we can adopt pre-training strategy. We train GMF and MLP and then use their model parameters as the initialization for the corresponding parts of NeuMF’s parameters. To get better performance of NeuMF, we can adopt pre-training strategy. We train GMF and MLP and then use their model parameters as the initialization for the corresponding parts of NeuMF’s parameters. 

We then evaluate pre-trained model using the same evaluation metrics. Compared with not pre-trained NMF, all evaluation metrics had slight improvements. The performance of pre-trained NCF is better than the not pre-trained.



## LSTM
(Step4D_LSTM.ipynb)
This model can only be run locally as there will be RAM limit error on Colab.

We explore sequential recommendation. A sequence length of 3 is specified, which means using sequences of two previous songs listened to by a user to predict the next song.

Sequence and Label Creation: For each user, sort their listening events by timestamp and create sequences of artist-track encodings along with their corresponding time-of-day features. Each sequence (excluding the last element) serves as input, and the next song (the last element in the sequence) is the label/target.
Input Sequence: Comprises the label-encoded artist-track IDs for two consecutive songs, concatenated with the one-hot encoded time features for those songs, flattened into a single vector.
Label: The label-encoded artist-track ID for the song following the sequence.

Split the structured sequences into training and test sets, ensuring that the sequences are not shuffled (shuffle=False). This is important for time series data to maintain the temporal order.

The input sequences are split into two parts: one containing the artist-track features (first two elements of each sequence) and another containing the time-of-day features (the rest of the sequence).
This separation facilitates different handling or processing paths in the neural network model, allowing for specialized layers (e.g., an embedding layer for artist-track features and a dense layer for time features).

### Model Specification
set_global_policy('mixed_float16'): This configures TensorFlow to use mixed precision training, which combines float32 and float16 data types to improve performance and reduce memory usage without compromising the model's accuracy. This significantly accelerates training on compatible hardware (GPUs with tensor cores).

EarlyStopping: A callback to stop training when a monitored metric has stopped improving, preventing overfitting. Here, we monitor validation loss ('val_loss'), with a patience of 10 epochs (i.e., training will stop if there is no improvement in validation loss for 10 consecutive epochs). 

Embedding Layer: Maps the integer-encoded artist-track IDs to dense vectors. This layer helps the model to understand the relationships between different IDs by projecting them into a continuous vector space.

LSTM Layer: Processes the sequences of embeddings with 40 units, using dropout and recurrent dropout to prevent overfitting.

Dense Time Features Layer: A dense layer that processes the one-hot encoded time features.

Output Layer: The final dense layer uses softmax activation to output a probability distribution over all possible artist-track IDs, corresponding to the model's prediction of the next song.

The model is compiled with the sparse_categorical_crossentropy loss function. The Adam optimizer is used with a learning rate of 0.001.

Determining Top-k Classes: The model's predictions are probabilities for each class. To evaluate its performance in a top-k context, we are interested in the classes with the highest probabilities. np.argsort(y_pred_prob, axis=1)[:, -k:][:, ::-1] sorts the classes by their predicted probabilities and selects the top 30 for each example. The [:, ::-1] part reverses the columns because np.argsort returns them in ascending order, and you want the highest probabilities first.

Transforming Indices to Class Names: label_encoder.inverse_transform(top_k_indices.flatten()).reshape(top_k_indices.shape) takes these top-k indices and converts them back into the original class labels (song names or IDs) using the inverse of the label encoding applied earlier.

## GCN Exploration
(Step4E_GCN_Exploration.ipynb)
Before running GCN models on the full dataset which has a time cost, we explore different model specifications of this model to find an optimal choice for later.

We construct the graph and provide visualizations for graph data. 

A train/validation/test split is used to achieve this by dividing the dataset into three sets: the training set, used to train the model; the validation set, used to tune the model's hyperparameters and prevent overfitting; and the test set, used to evaluate the model's performance on unseen data. We use a 70%-15%-15% split. For graph, we found a useful PyG method `RandomLinkSplit`, which works by randomly removing a specified percentage of the edges in the graph. The split is performed such that the training split doesn't include edges in the validation and test splits; and the validation split doesn't include edges in the test split. Also, since we plan to implement our own negative samplign algorithm we set add_negative_train_samples and neg_sampling_ratio to zero.

### LightGCN
To implement our model, we will be using the LightGCN architecture. This architecture forms a simple GNN method where we remove nonlinearity across layers. This leads to a highly scalable architecture with fewer parameters. By taking a weighted sum of the embeddings at different hop scales (also called multi-scale diffusion), LightGCN has exhibited better performance than other neural graph collaborative filtering approaches while also being computationally efficient. To implement our models, we will customize the implementation of LightGCN from PyG.

One important note: the GNN method we are defining below acts as our full graph neural network, consisting of multiple message passing layers that are connected with skip connections (weighted according to the alpha parameter). We surface functionality to change the message passing layer from the default LightGCN layer to alternatives, such as a GAT and GraphSAGE convolution instead, as well as to have a learnable alpha parameter.

The three convolutional layers we use are the LGConv (from LightGCN), SAGEConv (GraphSAGE), and GATConv (GAT). We add a linear layer on top of the GATConv to take the concatenated outputs from the multiple attention heads back to the embedding dimension. Below we provide the update steps for each type of layer.

1. LGConv

$$\begin{equation*}
\mathbf{e}_i^{(k+1)} = \underset{j \in \mathcal{N}(i)}{\sum} \frac{1}{
  \sqrt{| \mathcal{N}(i)|} \sqrt{| \mathcal{N}(j)|} } \mathbf{e}_j^{(k)}
\end{equation*}$$


2. SAGEConv

$$\begin{equation*}
\mathbf{e}^{(k+1)}_{i} = \mathbf{W}_1 \mathbf{e}^{(k)}_{i} + \mathbf{W}_2 \frac{1}{| \mathcal{N}(i)|} \underset{j \in \mathcal{N}(i)}{\sum} \mathbf{e}^{(k)}_j
\end{equation*}$$

3. GATConv

$$\begin{align*}
    \mathbf{e}^{(k+1)}_i &= \mathbf{\Theta}\mathbf{x}_i^{(k+1)} + \mathbf{B} \\
    \mathbf{x}_i^{(k+1)} &= \underset{h=1}{\Big\Vert^H} \sum_{j \in \mathcal{N}(i) \cup \{i\} } \alpha_{ij}^h \mathbf{W}^h
    \mathbf{e}^{(k)}_j \\
    \alpha^h_{ij} &= \frac{
    \exp(
    \text{LeakyReLU}(\mathbf{a}^{h^{T}} \left[\mathbf{W}^h \mathbf{e}_i \Vert \mathbf{W}^h \mathbf{e}_j \right]))
    }{
    \underset{l \in \mathcal{N}(i) \cup \{i\}}{\sum}     \exp(
    \text{LeakyReLU}(\mathbf{a}^{h^{T}} \left[\mathbf{W}^h \mathbf{e}_i \Vert \mathbf{W}^h \mathbf{e}_l \right]))
    }
\end{align*}$$


No matter which convolutional layer we use, we still take the weighted sum of the different layers as is standard in LightGCN. We do so as follows:

$$\begin{equation*}
    \mathbf{e}_i = \sum_{k=1}^K \alpha_k \mathbf{e}^{(k)}_i
\end{equation*}$$


Our main specifications will use a Bayesian Personalized Ranking, which is calculated as

$$\begin{equation*}
    \text{BPR Loss}(i) = \frac{1}{|\mathcal{E}(i)|} \underset{{(i, j_{+}) \in \mathcal{E}(i)}}{\sum} \log \sigma \left( \text{score}(i, j_+) - \text{score}(i, j_-) \right)
\end{equation*}$$

for a pair of positive edge $(i, j_{+})$ and negative edge $(i, j_{-})$. More on how we define a negative edge later.

### Negative Sampling
Important to any link prediction task is negative sampling. In the graph, we observe positive edges, which allows us to capture which nodes should be most similar to one another. Adding negative edges allows the model to explicitly capture that nodes that don't share an edge should have different embeddings. Without negative edges, you can convince yourself that a valid loss minimization strategy would be to simply assign all nodes the same embedding, which is obviously not meaningful or desirable.

Consequently, in this section, we define our negative sampling strategy. In particular, we take three approaches:
1. Random, no positive check: for each positive edge coming from a playlist $p_i$, randomly draw a track $t_j$ from the full set of track nodes such that ($p_i$, $t_j$) is the negative edge. For computational efficiency, we don't check if ($p_i$, $t_j$) is actually a negative edge, though probabilistically it is very likely.
2. Random, positive check: for each positive edge coming from a playlist $p_i$, randomly draw a track $t_j$ from the full set of track nodes such that ($p_i$, $t_j$) is the negative edge. We ensure that ($p_i$, $t_j$) is not a positive edge.
3. Hard: for each positive edge coming from a playlist $p_i$, randomly draw a track $t_j$ from the top $k$ proportion of tracks, ranked by dot product similarity to $p_i$. For epoch 0, $k = 1$ and we lower it at each subsequent iteration.

Since the result from the model with the 'LGC' convolutional layer is the best, we will be using that for the full dataset. In addition, to uniform the evaluation metrics, we will be using recall@30 instead. The full model code can be found in Step4F_GCN_full.ipynb file


## GCN on Full Dataset
(Step4F_GCN_Full.ipynb)
This model can only be run locally as there will be RAM limit error on Colab.

As mentioned in the GCN_model_comparison_on_smaller_dataset.ipynb, this file will be running a GCN model with 'LGC', lightGCN layer, on the full dataset and using K=30 for prediction of k songs.

We first constructing the nodes for tracks , users with their attributes. For each user, we get the list of their top songs, which is a track class object. We construct the graph & split the data into train-validation-test set (0.7-0.15-0.15). We then construct the GCN model. Do note that for our final result, only the model with 'LGC' LightGCN convolutional layer is run on the entire dataset due to time and computation power constraints. Other convolutional layers were explored on a subset of data only. 

Our main specifications will use a Bayesian Personalized Ranking, which is calculated as

$$\begin{equation*}
    \text{BPR Loss}(i) = \frac{1}{|\mathcal{E}(i)|} \underset{{(i, j_{+}) \in \mathcal{E}(i)}}{\sum} \log \sigma \left( \text{score}(i, j_+) - \text{score}(i, j_-) \right)
\end{equation*}$$

for a pair of positive edge $(i, j_{+})$ and negative edge $(i, j_{-})$. More on how we define a negative edge later.

Since our model focuses on the prediction of link between user and track node, a negative edge means that there is no link between such two nodes.

Important to any link prediction task is negative sampling. In the graph, we observe positive edges, which allows us to capture which nodes should be most similar to one another. Adding negative edges allows the model to explicitly capture that nodes that don't share an edge should have different embeddings. Without negative edges, you can convince yourself that a valid loss minimization strategy would be to simply assign all nodes the same embedding, which is obviously not meaningful or desirable.

Consequently, in this section, we define our negative sampling strategy. In particular, we take three approaches:
1. Random, no positive check: for each positive edge coming from a playlist $p_i$, randomly draw a track $t_j$ from the full set of track nodes such that ($p_i$, $t_j$) is the negative edge. For computational efficiency, we don't check if ($p_i$, $t_j$) is actually a negative edge, though probabilistically it is very likely.
2. Random, positive check: for each positive edge coming from a playlist $p_i$, randomly draw a track $t_j$ from the full set of track nodes such that ($p_i$, $t_j$) is the negative edge. We ensure that ($p_i$, $t_j$) is not a positive edge.
3. Hard: for each positive edge coming from a playlist $p_i$, randomly draw a track $t_j$ from the top $k$ proportion of tracks, ranked by dot product similarity to $p_i$. For epoch 0, $k = 1$ and we lower it at each subsequent iteration.

### GCN Evaluation
(Step4G_GCN_Evaluation.ipynb)
The evaluation metrics on top of loss calculation: recall at K.For a playlist $i$, $P^k_i$ represents the set of the top $k$ predicted tracks for $i$ and $R_i$ the ground truth of connected tracks to playlist $i$, then we calculate
$$
\text{recall}^k_i = \frac{| P^k_i \cap R_i | }{|R_i|}.
$$
If $R_i = 0$, then we assign this value to 1. Note, if $R_i \subset P_i^k$, then the recall is equal to 1. Hence, our choice of $k$ matters a lot.

Note: when evaluating this metric on our validation or test set, we need to make sure to filter the message passing edges from consideration, as the model can directly observe these.

We choose a value of $k = 30$ in this case, as each user has 50 songs in their top songs.

Since running the LightGCN model on the full dataset did not include the metrics of diversity, we evaluated such metrics by retrieving the final embedding learnt by the LightGCN model and use them to predict the top songs at k =30, and generate relevant metrics and visualising the evaluation.


# Evaluation of Models
We generally used the following evaluation metrics for all models developed:

- Ranking Metrics: These are used to evaluate how relevant recommendations are for users

MAP - It is the average precision for each user normalized over all users.

Normalized Discounted Cumulative Gain (NDCG) - evaluates how well the predicted items for a user are ranked based on relevance

Precision - this measures the proportion of recommended items that are relevant

Recall - this measures the proportion of relevant items that are recommended


- Non accuracy based metrics: These do not compare predictions against ground truth but instead evaluate the following properties of the recommendations

Novelty - measures of how novel recommendation items are by calculating their recommendation frequency among users

Diversity - measures of how different items in a set are with respect to each other

Serendipity - measures of how surprising recommendations are to to a specific user by comparing them to the items that the user has already interacted with

Coverage - measures related to the distribution of items recommended by the system.