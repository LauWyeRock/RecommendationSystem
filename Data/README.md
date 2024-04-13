# Data Collection, Preprocessing, Feature Engineering, Cleaning, and EDA Steps

This file introduces the steps we have taken to prepare data for ML models, the steps are as follows:

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


Feature Descriptions:
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