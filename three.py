import streamlit as st
from streamlit_card import card
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Function to load dataframes
def load_data(file):
    return pd.read_csv(file)

# Upload CSV files
st.title("Cricket Player Analysis")

bat_main_file = st.file_uploader("Upload Batting Main CSV", type=["csv"])
bat_avg_file = st.file_uploader("Upload Batting Average CSV", type=["csv"])
bat_sixes_file = st.file_uploader("Upload Batting Sixes CSV", type=["csv"])
bat_sr_file = st.file_uploader("Upload Batting Strike Rate CSV", type=["csv"])
bowling_wickets_file = st.file_uploader("Upload Bowling Wickets CSV", type=["csv"])
bowling_avg_file = st.file_uploader("Upload Bowling Average CSV", type=["csv"])
bowling_econ_file = st.file_uploader("Upload Bowling Economy CSV", type=["csv"])

if bat_main_file and bat_avg_file and bat_sixes_file and bat_sr_file and \
   bowling_wickets_file and bowling_avg_file and bowling_econ_file:

    # Load data
    bat_main = load_data(bat_main_file)
    bat_avg = load_data(bat_avg_file)
    bat_sixes = load_data(bat_sixes_file)
    bat_sr = load_data(bat_sr_file)
    bowling_wickets = load_data(bowling_wickets_file)
    bowling_avg = load_data(bowling_avg_file)
    bowling_econ = load_data(bowling_econ_file)

    # Filter players with more than 100 innings
    bat_main_1 = bat_main[bat_main['Inns'] > 100]
    bat_avg_1 = bat_avg[bat_avg['Inns'] > 100]
    bat_sixes_1 = bat_sixes[bat_sixes['Inns'] > 100]
    bat_sr_1 = bat_sr[bat_sr['Inns'] > 100]
    bowling_wickets_1 = bowling_wickets[bowling_wickets['Mat'] > 100]
    bowling_avg_1 = bowling_avg[bowling_avg['Mat'] > 100]
    bowling_econ_1 = bowling_econ[bowling_econ['Mat'] > 100]

    # Merge batting dataframes
    bat_df = (bat_main_1.merge(bat_avg_1, on='Player', how='inner', suffixes=('', '_avg'))
              .merge(bat_sixes_1, on='Player', how='inner', suffixes=('', '_sixes'))
              .merge(bat_sr_1, on='Player', how='inner', suffixes=('', '_sr')))

    # Merge bowling dataframes
    bowl_df = (bowling_wickets_1.merge(bowling_avg_1, on='Player', how='inner', suffixes=('', '_avg'))
               .merge(bowling_econ_1, on='Player', how='inner', suffixes=('', '_econ')))

    # Add hypothetical target columns
    bat_df['Class'] = np.random.choice(['Type1', 'Type2'], len(bat_df))
    bowl_df['Class'] = np.random.choice(['Type1', 'Type2'], len(bowl_df))

    # Prepare features and targets for batting
    X_bat = bat_df[['Ave_avg', 'Runs', 'SR']]
    y_bat = bat_df['Class']

    # Prepare features and targets for bowling
    X_bowl = bowl_df[['Ave_avg', 'Wkts', 'Econ']]
    y_bowl = bowl_df['Class']

    # Split data into training and testing sets
    X_bat_train, X_bat_test, y_bat_train, y_bat_test = train_test_split(X_bat, y_bat, test_size=0.2, random_state=42)
    X_bowl_train, X_bowl_test, y_bowl_train, y_bowl_test = train_test_split(X_bowl, y_bowl, test_size=0.2, random_state=42)

    # Train models
    rf_bat = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_bowl = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_bat.fit(X_bat_train, y_bat_train)
    rf_bowl.fit(X_bowl_train, y_bowl_train)

    # Predictions
    y_bat_pred = rf_bat.predict(X_bat_test)
    y_bowl_pred = rf_bowl.predict(X_bowl_test)

    # Display results
    st.subheader("Batting Performance")
    st.text(classification_report(y_bat_test, y_bat_pred))
    st.text(f"Accuracy: {accuracy_score(y_bat_test, y_bat_pred)}")

    st.subheader("Bowling Performance")
    st.text(classification_report(y_bowl_test, y_bowl_pred))
    st.text(f"Accuracy: {accuracy_score(y_bowl_test, y_bowl_pred)}")

    # Selecting best players
    # Best batsmen
    best_batsmen = pd.concat([
        bat_df.nlargest(2, 'Ave'),
        bat_df.nlargest(2, 'Runs'),
        bat_df.nlargest(2, 'SR')  # Removed 'Away' condition for now
    ]).drop_duplicates()

    # Best bowlers
    best_bowlers = pd.concat([
        bowl_df.nlargest(2, 'Wkts'),
        bowl_df.nsmallest(2, 'Econ'),
        bowl_df.nsmallest(1, 'Ave_avg')
    ]).drop_duplicates()

    # Select top 5 bowlers
    best_bowlers = best_bowlers.head(5)

    # Combine best batsmen and bowlers
    best_players = pd.concat([best_batsmen, best_bowlers])

    # Display the best players
    st.subheader("Best Playing XI")
    st.write(best_players[['Player']])


    st.subheader("Best IPL Playing XI")

    # Create grid of cards
    col1, col2, col3 = st.columns(3)

    with col1:
        card(
            title="",
            text="",
            image="https://documents.iplt20.com/ipl/IPLHeadshot2024/214.png",
            url="https://www.iplt20.com/teams/delhi-capitals/squad-details/170",
            styles={
            "card": {
            "width": "200px",
            "height": "200px",
            "border-radius": "10px",
            "box-shadow": "0 0 10px rgba(0,0,0,0.5)",},
            "filter": {
            "background-color": "rgba(255, 99, 71, 0)",}}
        )
        
        # Use HTML inside st.markdown
        st.markdown("<h1 style='font-size: 36px;'>Warner + Batsmen</h1>", unsafe_allow_html=True)

        card(
            title="",
            text="",
            image="https://documents.iplt20.com/ipl/IPLHeadshot2024/19.png",
            url="https://www.iplt20.com/teams/lucknow-super-giants/squad-details/1125",
            styles={
            "card": {
            "width": "200px",
            "height": "200px",
            "border-radius": "10px",
            "box-shadow": "0 0 10px rgba(0,0,0,0.5)",},
            "filter": {
            "background-color": "rgba(255, 99, 71, 0)",}}
        )

        st.markdown("<h1 style='font-size: 36px;'>KL Rahul  Batsmen</h1>", unsafe_allow_html=True)
        
        card(
            title="",
            text="",
            image="https://documents.iplt20.com/ipl/IPLHeadshot2024/156.png",
            url="https://www.iplt20.com/teams/kolkata-knight-riders/squad-details/203",
            styles={
            "card": {
            "width": "200px",
            "height": "200px",
            "border-radius": "10px",
            "box-shadow": "0 0 10px rgba(0,0,0,0.5)",},
            "filter": {
            "background-color": "rgba(255, 99, 71, 0)",}}
        )

        st.markdown("<h1 style='font-size: 36px;'>S Naraine + <br> All-Rounder</h1>", unsafe_allow_html=True)

        card(
        title="",
        text="",
        image="https://documents.iplt20.com/playerheadshot/ipl/284/211.png",
        url="https://www.iplt20.com/teams/mumbai-indians/squad-details/211",
        styles={
            "card": {
            "width": "200px",
            "height": "200px",
            "border-radius": "10px",
            "box-shadow": "0 0 10px rgba(0,0,0,0.5)",},
            "filter": {
            "background-color": "rgba(255, 99, 71, 0)",}}
        )

        st.markdown("<h1 style='font-size: 36px;'>Malinga + <br> Bowler</h1>", unsafe_allow_html=True)

    with col2:
        card(
            title="",
            text="",
            image="https://documents.iplt20.com/ipl/IPLHeadshot2024/11.png",
            url="https://www.iplt20.com/teams/punjab-kings/squad-details/41",
            styles={
            "card": {
            "width": "200px",
            "height": "200px",
            "border-radius": "10px",
            "box-shadow": "0 0 10px rgba(0,0,0,0.5)",},
            "filter": {
            "background-color": "rgba(255, 99, 71, 0)",}}
        )

        st.markdown("<h1 style='font-size: 36px;'>Dhawan Batsmen</h1>", unsafe_allow_html=True)

        card(
            title="",
            text="",
            image="https://documents.iplt20.com/ipl/IPLHeadshot2024/18.png",
            url="https://www.iplt20.com/teams/delhi-capitals/squad-details/2972",
            styles={
            "card": {
            "width": "200px",
            "height": "200px",
            "border-radius": "10px",
            "box-shadow": "0 0 10px rgba(0,0,0,0.5)",},
            "filter": {
            "background-color": "rgba(255, 99, 71, 0)",}}
        )
        
        st.markdown("<h1 style='font-size: 36px;'>Pant  WicketKeeper</h1>", unsafe_allow_html=True)

        card(
            title="",
            text="",
            image="https://documents.iplt20.com/ipl/IPLHeadshot2024/218.png",
            url="https://www.iplt20.com/teams/gujarat-titans/squad-details/2885",
            styles={
            "card": {
            "width": "200px",
            "height": "200px",
            "border-radius": "10px",
            "box-shadow": "0 0 10px rgba(0,0,0,0.5)",},
            "filter": {
            "background-color": "rgba(255, 99, 71, 0)",}}
        )

        st.markdown("<h1 style='font-size: 36px;'>R Khan + Bowler</h1>", unsafe_allow_html=True)

        card(
        title="",
        text="",
        image="https://documents.iplt20.com/ipl/IPLHeadshot2024/9.png",
        url="https://www.iplt20.com/teams/mumbai-indians/squad-details/1124",
        styles={
            "card": {
            "width": "200px",
            "height": "200px",
            "border-radius": "10px",
            "box-shadow": "0 0 10px rgba(0,0,0,0.5)",},
            "filter": {
            "background-color": "rgba(255, 99, 71, 0)",}}
        )

        st.markdown("<h1 style='font-size: 36px;'>Bumrah  Bowler</h1>", unsafe_allow_html=True)

        

    with col3:
        card(
            title="",
            text="",
            image="https://documents.iplt20.com/ipl/IPLHeadshot2024/2.png",
            url="https://www.iplt20.com/teams/royal-challengers-bangalore/squad-details/164",
            styles={
            "card": {
            "width": "200px",
            "height": "200px",
            "border-radius": "10px",
            "box-shadow": "0 0 10px rgba(0,0,0,0.5)",},
            "filter": {
            "background-color": "rgba(255, 99, 71, 0)",}}
        )
        
        st.markdown("<h1 style='font-size: 36px;'>Kohli  Batsmen</h1>", unsafe_allow_html=True)

        card(
            title="",
            text="",
            image="https://documents.iplt20.com/ipl/IPLHeadshot2024/54.png",
            url="https://www.iplt20.com/teams/mumbai-indians/squad-details/2740",
            styles={
            "card": {
            "width": "200px",
            "height": "200px",
            "border-radius": "10px",
            "box-shadow": "0 0 10px rgba(0,0,0,0.5)",},
            "filter": {
            "background-color": "rgba(255, 99, 71, 0)",}}
        )

        st.markdown("<h1 style='font-size: 36px;'>H Pandya <br> All Rounder</h1>", unsafe_allow_html=True)     

        card(
            title="",
            text="",
            image="https://documents.iplt20.com/ipl/IPLHeadshot2024/45.png",
            url="https://www.iplt20.com/teams/rajasthan-royals/squad-details/8",
            styles={
            "card": {
            "width": "200px",
            "height": "200px",
            "border-radius": "10px",
            "box-shadow": "0 0 10px rgba(0,0,0,0.5)",},
            "filter": {
            "background-color": "rgba(255, 99, 71, 0)",}}
        )

        st.markdown("<h1 style='font-size: 36px;'>R Ashwin <br> All Rounder</h1>", unsafe_allow_html=True)
    

    # Save best players to Excel
    best_players[['Player']].to_excel('best_players.xlsx', index=False)
