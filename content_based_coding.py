import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from dataset import read_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class recommendation_based_one_content:

    # Load the dataset
    data = read_data.load_data()

    def data_clean(data):
        data.drop('ratings', inplace=True, axis=1)
        data.duplicated(subset='name').sum()
        data.drop_duplicates(subset='name', inplace=True)
        data.dropna(inplace=True)
        return data
    
    data = data_clean(data)

    # Convert all strings to lower
    # oppo not equal Oppo >>>>>>>>> so the function will calculate it 2 times and we do not need that
    def clean_data(x):
        return str.lower(x)
            
    # applying data clean function on data
    data['name'] = data['name'].apply(clean_data) 

    def convert_data_into_vectors(data):
        # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
        tf = TfidfVectorizer(stop_words='english')
        # Construct the required TF-IDF matrix by fitting and transforming the data
        tfidf_matrix = tf.fit_transform(data['name'])
        return tfidf_matrix

    def calculate_cosin_similarity(tfidf_matrix):
        # Calculate Cosin Similarity
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        return cosine_sim
    
    def matching_index_with_matrix(data):
        # For matching the name with index in cosin matrix
        data = data.reset_index()
        names = data['name']
        indices = pd.Series(data.index, index=data['name'])
        return indices, names

    tf_matrix = convert_data_into_vectors(data=data)
    cosin_matrix = calculate_cosin_similarity(tfidf_matrix=tf_matrix)
    index, names = matching_index_with_matrix(data=data)

    # Get Recommendation Data
    def get_recommendations(name, number_top, cosine_sim=cosin_matrix, indices=index, data=data):
        idx = indices[name]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[0:number_top]
        phone_indices = [i[0] for i in sim_scores]
        # phone_scores = [i[1] for i in sim_scores]
        # dic = {'phone_indices':phone_indices, 'phone_scores':phone_scores}
        # df = pd.DataFrame(dic)
        result = data[['imgURL','name','price','corpus']].iloc[phone_indices]
        final_data = pd.DataFrame(result).drop_duplicates()
        return final_data

class recommendation_based_many_content:

    data = read_data.load_data()

    def data_clean(data):
        data.drop('ratings', inplace=True, axis=1)
        data.duplicated(subset='name').sum()
        data.drop_duplicates(subset='name', inplace=True)
        data.dropna(inplace=True)
        return data
    
    data = data_clean(data)

    # Content based many columns like name and corpus
    features = ['name', 'corpus']

    # Convert all strings to lower
    # oppo not equal Oppo >>>>>>>>> so the function will calculate it 2 times and we do not need that
    def clean_data(x):
        return str.lower(x)
            
    # applying data clean function on data
    for feature in features:
        data[feature] = data[feature].apply(clean_data)

    # Soup the Data
    data['Features'] = data['name'] + ',' + data['corpus'] 

    # convert to vectors
    def convert_to_vector(data):
        count = CountVectorizer(stop_words='english')
        count_matrix2 = count.fit_transform(data['Features'])
        return count_matrix2
    
    # Calculate Cosin Similarity
    def calculate_cosin_similarity(count_matrix2):
        cosine_sim2 = cosine_similarity(count_matrix2, count_matrix2)
        return cosine_sim2
    
    # For matching the name with index in cosin matrix
    def matching_index_with_matrix(data):
        # For matching the name with index in cosin matrix
        data = data.reset_index()
        names = data['name']
        indices = pd.Series(data.index, index=data['name'])
        return indices, names

    tf_matrix = convert_to_vector(data=data)
    cosin_matrix = calculate_cosin_similarity(count_matrix2=tf_matrix)
    index, names = matching_index_with_matrix(data=data)

    # Get Recommendation Data
    def get_recommendations(name, number_top, cosine_sim=cosin_matrix, indices=index, data=data):
        idx = indices[name]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[0:number_top]
        phone_indices = [i[0] for i in sim_scores]
        # phone_scores = [i[1] for i in sim_scores]
        # dic = {'phone_indices':phone_indices, 'phone_scores':phone_scores}
        # df = pd.DataFrame(dic)
        result = data[['imgURL','name','price','corpus']].iloc[phone_indices]
        final_data = pd.DataFrame(result).drop_duplicates()
        return final_data

    
