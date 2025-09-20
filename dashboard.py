import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# 🎨 Styling
st.markdown("""
    <style>
    .main {
        background-color: #F5F9FF;
        color: #000000;
    }
    .block-container {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 12px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1B4F72;
        cursor: pointer;
    }
    .sidebar .sidebar-content {
        background-color: #EBF3FA;
    }
    a {
        color: #0B5345;
        text-decoration: none;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
rf_model = joblib.load("rf_model.pkl")
svm_model = joblib.load("svm_model.pkl")
knn_model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")


st.sidebar.markdown("### 🗳️ Quick Poll: What's your market outlook?")

poll_options = ["Bullish (Expect prices to rise)", 
                "Bearish (Expect prices to fall)", 
                "Neutral (No clear direction)"]

# Initialize vote counts in session_state
if 'poll_votes' not in st.session_state:
    st.session_state['poll_votes'] = {option: 0 for option in poll_options}

# User vote
user_choice = st.sidebar.radio("Select your sentiment:", poll_options, key="market_poll")

# Submit button to register vote
if st.sidebar.button("Submit Vote"):
    st.session_state['poll_votes'][user_choice] += 1
    st.sidebar.success(f"Thanks for voting: **{user_choice}**")

# Calculate total votes
total_votes = sum(st.session_state['poll_votes'].values())

# Show results with percentages
st.sidebar.markdown("#### Poll Results:")
for option, count in st.session_state['poll_votes'].items():
    pct = (count / total_votes * 100) if total_votes > 0 else 0
    st.sidebar.write(f"{option}: {count} votes ({pct:.1f}%)")


# Sidebar content
st.sidebar.title("📌 Market Insights")

st.sidebar.markdown("### 🌐 US Stock Market News")
st.sidebar.markdown("[CNBC Markets](https://www.cnbc.com/markets/)")
st.sidebar.markdown("[Yahoo Finance](https://finance.yahoo.com/)")
st.sidebar.markdown("[MarketWatch](https://www.marketwatch.com/)")
st.sidebar.markdown("[Bloomberg Markets](https://www.bloomberg.com/markets)")
st.sidebar.markdown("[Reuters Business](https://www.reuters.com/markets/)")

# Sentiment Analysis moved to sidebar
st.sidebar.markdown("### 📰 Sentiment Analysis on News Headlines")
user_text = st.sidebar.text_area(
    "Paste recent news headlines or comments about the stock (one per line)", 
    key="sentiment_input"
)

if user_text:
    lines = user_text.strip().split('\n')
    sentiments = []
    for line in lines:
        analysis = TextBlob(line)
        polarity = analysis.sentiment.polarity
        sentiments.append(polarity)
    avg_sentiment = np.mean(sentiments)
    st.sidebar.write(f"**Average Sentiment Polarity:** {avg_sentiment:.3f}")
    if avg_sentiment > 0.1:
        st.sidebar.success("Overall positive sentiment detected 👍")
    elif avg_sentiment < -0.1:
        st.sidebar.error("Overall negative sentiment detected 👎")
    else:
        st.sidebar.info("Neutral or mixed sentiment detected 🤔")

st.title("📈 Stock Price Movement Prediction Dashboard")

uploaded_file = st.file_uploader("Upload Stock Data (.csv or .xlsx)", type=["csv", "xlsx"], key="stock_data")

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)

    # Map stock name from filename prefix
    stock_name_map = {"aap": "Apple", "msft": "Microsoft", "goog": "Google", "amzn": "Amazon", "tsla": "Tesla"}
    file_prefix = uploaded_file.name.split('.')[0].split('_')[0].lower()
    stock_name = stock_name_map.get(file_prefix, file_prefix.upper())
    st.subheader(f"📊 Stock: {stock_name}")

    st.write("### 📄 Raw Data", df.head())

    # Indicators
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Targets
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()

    # Features
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    X = df[feature_cols]
    y = df['Target']
    X_scaled = scaler.transform(X)

    # Model selection with unique key
    model_option = st.selectbox("Choose Model", ("Random Forest", "SVM", "KNN"), key="model_select")
    model = rf_model if model_option == "Random Forest" else svm_model if model_option == "SVM" else knn_model
    preds = model.predict(X if model_option == "Random Forest" else X_scaled)

    # Accuracy
    accuracy = accuracy_score(y, preds)
    df['Prediction'] = preds
    st.write(f"📊 **Model Accuracy:** {accuracy:.2%}")

    # SMA vs EMA Header with info popup
    if st.button("### 📉 Close Price with SMA & EMA", key="sma_ema_header"):
        st.info("""
| Condition                 | Action |
| ------------------------- | ------ |
| EMA crosses **above** SMA | Buy    |
| EMA crosses **below** SMA | Sell   |
        """)

    # Plot SMA and EMA
    fig1, ax1 = plt.subplots()
    df[['Close', 'SMA_10', 'EMA_10']].tail(200).plot(ax=ax1)
    ax1.set_ylabel("Price")
    ax1.set_xlabel("Index")
    ax1.set_title("Close Price with SMA & EMA")
    st.pyplot(fig1)

    # RSI Header with info popup
    if st.button("### 📊 RSI Indicator", key="rsi_header"):
        st.info("""
| RSI Level               | Signal        |
| ----------------------- | ------------- |
| RSI > 70                | Overbought - Sell |
| RSI < 30                | Oversold - Buy   |
| RSI between 30 and 70   | Hold/Neutral     |
        """)

    # Plot RSI
    fig2, ax2 = plt.subplots()
    df['RSI_14'].tail(200).plot(ax=ax2, color='purple')
    ax2.axhline(70, color='red', linestyle='--')
    ax2.axhline(30, color='green', linestyle='--')
    ax2.set_ylabel("RSI")
    ax2.set_title("Relative Strength Index (14)")
    st.pyplot(fig2)

    # Cumulative Returns Chart
    st.write("### 💰 Cumulative Returns from Model Predictions")
    df['Strategy_Returns'] = df['Target'] * df['Close'].pct_change().shift(-1)
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns'].fillna(0)).cumprod() - 1
    df['Cumulative_Market_Returns'] = (1 + df['Close'].pct_change().fillna(0)).cumprod() - 1

    fig3, ax3 = plt.subplots()
    df[['Cumulative_Strategy_Returns']].tail(300).plot(ax=ax3, color='blue')
    ax3.set_ylabel("Cumulative Returns")
    ax3.set_title("Model Strategy vs Market Returns")
    st.pyplot(fig3)

    # CSV Download
    csv = df[['Close', 'Prediction']].to_csv(index=False).encode()
    st.download_button("📥 Download Predictions CSV", csv, "predictions.csv", "text/csv")

    # Next Day Prediction
    st.write("### 🔮 Next Day Prediction")
    last_row = df[feature_cols].iloc[-1:]
    next_pred = model.predict(last_row if model_option == "Random Forest" else scaler.transform(last_row))[0]
    if next_pred == 1:
        st.success("📈 UP Prediction for next day!")
    else:
        st.error("📉 DOWN Prediction for next day.")
else:
    st.info("Upload a stock data file to get started.")
