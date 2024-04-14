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
We explore content-based filtering method. We use the 6 different Principal Components to find the top 5 most similar tracks using cosine similarity, denoted by 'angular' using the Annoy package. The output of this was then utlized in GCN models.

## KMeans
(Step3A_KMeans.ipynb)
We explore content-based filtering method. Songs are described with a set of lyrics and audio features, or the principal components generated during feature engineering. We assume users prefer to listen to songs that are similar to what they have listened to before. Given a song, we provide recommendation of top 30 most similar songs from the same cluster.

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


### LSTM
(Step4D_LSTM.ipynb)
We obtain 



# Evaluation of Models
We generally use the following evaluation metrics for all models developed:

- Ranking Metrics: These are used to evaluate how relevant recommendations are for users

MAP - It is the average precision for each user normalized over all users.

Normalized Discounted Cumulative Gain (NDCG) - evaluates how well the predicted items for a user are ranked based on relevance

Precision - this measures the proportion of recommended items that are relevant

Recall - this measures the proportion of relevant items that are recommended


- Rating Metrics: These are used to evaluate how accurate a recommender is at predicting ratings that users gave to items

Root Mean Square Error (RMSE) - measure of average error in predicted ratings

R Squared (R2) - essentially how much of the total variation is explained by the model

Mean Absolute Error (MAE) - similar to RMSE but uses absolute value instead of squaring and taking the root of the average

Explained Variance - how much of the variance in the data is explained by the model


- Non accuracy based metrics: These do not compare predictions against ground truth but instead evaluate the following properties of the recommendations

Novelty - measures of how novel recommendation items are by calculating their recommendation frequency among users

Diversity - measures of how different items in a set are with respect to each other

Serendipity - measures of how surprising recommendations are to to a specific user by comparing them to the items that the user has already interacted with

Coverage - measures related to the distribution of items recommended by the system.