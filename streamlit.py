import streamlit as st
st.set_page_config(
    page_title="Phones Recommendation System",
    page_icon=":iphone",
    layout="wide",
    initial_sidebar_state="expanded")


from content_based_coding import recommendation_based_one_content, recommendation_based_many_content

# Load the dataset
data = recommendation_based_one_content.data

# Create a Streamlit app
st.header(':iphone: :orange[Phones Recommendation System]', divider='rainbow')

# Display some random books
st.subheader('Some Random Phones:')
# st.dataframe(books.sample(5))
st.data_editor(
    data.sample(5),
    column_config={
        "imgURL": st.column_config.ImageColumn(
            "Preview Image", help="Streamlit app preview screenshots"
        )
    },
    use_container_width=True,
    hide_index=True,
    
)
st.divider()

with st.sidebar:
    st.header(':iphone: :orange[Content Based Filtering]', divider='rainbow')
    # User input for recommendation
    number_of_books = st.number_input('Enter a number of top Phones:', 1)
    name_phone = st.text_input('Enter the name of Phone:', 'oneplus 9 5g (winter mist, 128 gb)')
    # Get recommendations
    st.info('Filtering Based on one Vector', icon="ℹ️")
    btn_get_rec_vector = st.button('Get Recommendations', key=1)
    st.info('Filtering Based on many Vectors', icon="ℹ️")
    btn_get_rec_many_vectors = st.button('Get Recommendations', key=2)
    
if btn_get_rec_vector:
    name_lower = str.lower(name_phone)
    recommendations = recommendation_based_one_content.get_recommendations(name_lower, number_of_books)
    st.subheader('Recommended Phones:')
    st.data_editor(
        recommendations,
        column_config={
            "imgURL": st.column_config.ImageColumn(
                "Preview Image", help="Streamlit app preview screenshots"
            )
        },
        use_container_width=True,
        hide_index=True,
    )
try:    
    if btn_get_rec_many_vectors:
        name_lower = str.lower(name_phone)
        recommendations = recommendation_based_many_content.get_recommendations(name_lower, number_of_books)
        st.subheader('Recommended Phones based many content:')
        st.data_editor(
            recommendations,
            column_config={
                "imgURL": st.column_config.ImageColumn(
                    "Preview Image", help="Streamlit app preview screenshots"
                )
            },
            use_container_width=True,
            hide_index=True,
        )
except:
    st.error('Sorry! No Data Matching!', icon="🚨")
    st.warning('Try another phone!', icon="⚠️")