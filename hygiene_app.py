import nltk as nltk
nltk.download('vader_lexicon')
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random


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


def likelihood_hygiene_indicator(sentiment):
    # Define likelihood ranges based on sentiment
    if sentiment == "Positive":
        return random.randint(80, 95)
    elif sentiment == "Neutral":
        return random.randint(40, 60)
    else:
        return random.randint(0, 20)


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
    cuisine_options = ["Italian", "Mexican", "Indian", "Japanese", "American", "Chinese", "Thai", "Mediterranean"]
    location_options = ['98118', '98109', '98103', '98112', '98102', '98107', '98105', '98108', '98104', '98122']

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
        hygiene_likelihood = likelihood_hygiene_indicator(sentiment)
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
