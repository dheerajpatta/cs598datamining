import nltk as nltk
nltk.download('vader_lexicon')
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random

print('The nltk version is {}.'.format(nltk.__version__))

def sentiment_analysis(text):
    # Initialize the SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # Perform sentiment analysis using VaderSentiment
    sentiment_score = sia.polarity_scores(text)['compound']

    # Define sentiment categories
    if sentiment_score >= 0.05:
        return "Positive"
    elif -0.05 < sentiment_score < 0.05:
        return "Neutral"
    else:
        return "Negative"


def likelihood_hygiene_indicator(review_text, cuisine_type, zipcode, average_rating):
    cuisine_type = str(cuisine_type) # need to convert list to string
    data = [[review_text, cuisine_type, zipcode, average_rating]]
    input_df = pd.DataFrame(data, columns = ['text', 'cuisines_offered', 'zipcode', 'avg_rating'])
    hygiene_likelihood = float(pipeline_serialized.predict_probability(input_df)[:,1])
    # output_string = 'This restaurant is: {:.2%} likely to pass a hygiene inspection'.format(pred)
    return hygiene_likelihood

# Load the trained machine learning model
# Load serialized model
serialize_path = './pipeline.joblib'
pipeline_serialized = load(serialize_path) 

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Restaurant Hygiene Predictor",
        page_icon="🍔",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Define color palette
    primary_color = "#FF8C00"  # Dark Orange
    secondary_color = "#006400"  # Dark Green
    bg_color = "#F5F5F5"  # Light Gray

    # Add custom CSS for styling
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background-color: {bg_color};
        }}
        .sidebar .sidebar-content {{
            background-color: {secondary_color};
            color: #FFFFFF;
        }}
        .Widget>label {{
            color: {secondary_color};
            font-size: 20px;
            font-weight: bold;
        }}
        .stButton>button {{
            background-color: {primary_color};
            color: #FFFFFF;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Restaurant Hygiene Predictor")

    # Define the maximum number of characters allowed in the text box
    max_chars = 4000

    # Create a text area widget with a character limit
    st.markdown("<h3>Enter Review Text</h3>", unsafe_allow_html=True)
    review_text = st.text_area("Enter Review Text (4000 characters or less)", max_chars=max_chars, height=200)

    # Display the remaining character count
    remaining_chars = max_chars - len(review_text)
    st.markdown(f"<p>Remaining characters: {remaining_chars}</p>", unsafe_allow_html=True)

    st.markdown("---")

    # Cuisine options and location zip codes
    cuisine_options = ['Afghan', 'African', 'American (New)', 'American (Traditional)', 'Asian Fusion', 'Australian', 
                        'Barbeque', 'Basque', 'Belgian', 'Brazilian', 'Breakfast & Brunch', 'British', 'Buffets', 
                        'Burgers', 'Cafes', 'Cajun/Creole', 'Cambodian', 'Cantonese', 'Caribbean', 'Cheesesteaks', 
                        'Chicken Wings', 'Chinese', 'Colombian', 'Comfort Food', 'Creperies', 'Cuban', 'Delis', 
                        'Dim Sum', 'Diners', 'Egyptian', 'Ethiopian', 'Fast Food', 'Filipino', 'Fish & Chips', 'Fondue', 
                        'Food Court', 'Food Stands', 'French', 'Gastropubs', 'German', 'Gluten-Free', 'Greek', 'Haitian', 
                        'Halal', 'Hawaiian', 'Himalayan/Nepalese', 'Hot Dogs', 'Hot Pot', 'Indian', 'Indonesian', 'Irish', 
                        'Italian', 'Japanese', 'Korean', 'Kosher', 'Laotian', 'Latin American', 'Lebanese', 'Live/Raw Food', 
                        'Malaysian', 'Mediterranean', 'Mexican', 'Middle Eastern', 'Modern European', 'Mongolian', 'Moroccan', 
                        'Pakistani', 'Persian/Iranian', 'Pizza', 'Polish', 'Puerto Rican', 'Restaurants', 'Russian', 'Salad', 
                        'Salvadoran', 'Sandwiches', 'Scandinavian', 'Scottish', 'Seafood', 'Senegalese', 'Shanghainese', 
                        'Soul Food', 'Soup', 'Southern', 'Spanish', 'Steakhouses', 'Sushi Bars', 'Szechuan', 'Taiwanese', 
                        'Tapas Bars', 'Tapas/Small Plates', 'Tex-Mex', 'Thai', 'Trinidadian', 'Turkish', 'Vegan', 'Vegetarian', 
                        'Venezuelan', 'Vietnamese']
                        
    location_options = ['98118', '98109', '98103', '98112', '98102', '98107', '98105',
                        '98108', '98104', '98122', '98106', '98101', '98134', '98121',
                        '98199', '98146', '98115', '98125', '98119', '98144', '98126',
                        '98116', '98117', '98133', '98136', '98188', '98168', '98178',
                        '98166', '98177']

    # Select preferred Cuisine
    st.markdown("<h3>Select Preferred Cuisine</h3>", unsafe_allow_html=True)
    preferred_cuisine = st.selectbox("Choose Cuisine", cuisine_options)

    # Select preferred Location
    st.markdown("<h3>Select Preferred Location</h3>", unsafe_allow_html=True)
    preferred_location = st.selectbox("Choose Zip Code", location_options)

    # Select preferred Star Rating
    st.markdown("<h3>Select Preferred Star Rating</h3>", unsafe_allow_html=True)
    preferred_star_rating = st.radio("", [1, 2, 3, 4, 5], index=2)

    st.markdown("---")

    # Submit and Clear buttons with emojis as icons
    col1, col2 = st.columns(2)
    if col1.button("Submit 🚀"):
        predicting_message = st.info("Predicting...")
        # st.info("Predicting...")

        # Perform sentiment analysis on the entered review text
        sentiment = sentiment_analysis(review_text)

        # Get random likelihood of % hygiene indicator based on sentiment
        hygiene_likelihood = likelihood_hygiene_indicator(review_text, cuisine_type, zipcode, average_rating):
        
        # Set custom CSS for the output elements to have a green background

        # Show the result with a success message
        predicting_message.success("Success!")
        # st.success("Success!")
        st.write("Review Text:", review_text)
        st.write("Preferred Cuisine:", preferred_cuisine)
        st.write("Preferred Location (Zip Code):", preferred_location)
        st.write("Preferred Star Rating:", preferred_star_rating)
        st.write("**Sentiment:**", f"**{sentiment}**")
        st.success(f"**The restaurant has a likelihood of **{hygiene_likelihood}%** clearing a Hygiene Inspection**")

    if col2.button("Clear ❌"):
        # Clear the review text when the "Clear" button is clicked
        review_text = ""
        preferred_cuisine = cuisine_options[0]
        preferred_location = location_options[0]
        preferred_star_rating = 3

        # Reset the widgets with cleared values
        st.experimental_set_query_params()
        st.experimental_rerun()
        # Alternatively, use st.session_state to reset the inputs
        st.session_state.review_text = ""
        st.session_state.preferred_cuisine = cuisine_options[0]
        st.session_state.preferred_location = location_options[0]
        st.session_state.preferred_star_rating = 3


if __name__ == "__main__":
    main()
