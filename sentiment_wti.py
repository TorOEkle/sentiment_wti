import pandas as pd
import matplotlib.pyplot as plt
import logging
from functions import correlation_with_se
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    df = pd.read_csv("dallas_fed_NLP.csv")
    sentiment_map = {
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1
    }   

    df["SentimentScore"] = df["Sentiment"].map(sentiment_map)
    sentiment_by_time = df.groupby("Time")["SentimentScore"].mean().reset_index()
    sentiment_by_time["Time"] = pd.PeriodIndex(sentiment_by_time["Time"], freq="Q")
    sentiment_by_time = sentiment_by_time.sort_values("Time")

    oil_price = pd.read_csv("DCOILWTICO.csv")
    oil_price["observation_date"] = pd.to_datetime(oil_price["observation_date"])
    oil_price = oil_price.dropna(subset=["DCOILWTICO"])
    oil_price = oil_price[oil_price["observation_date"] >= "2017-01-01"]
    oil_price["Quarter"] = oil_price["observation_date"].dt.to_period("Q")
    quarterly_oil = (
        oil_price.groupby("Quarter")["DCOILWTICO"]
        .mean()
        .reset_index()
    )
    quarterly_oil["Time"] = quarterly_oil["Quarter"].astype(str)
    quarterly_oil = quarterly_oil.drop(columns=["Quarter"])
    quarterly_oil = quarterly_oil.rename(columns={"DCOILWTICO": "WTI"})
    quarterly_oil["Time"] = pd.PeriodIndex(quarterly_oil["Time"], freq="Q")

    sentiment_oil_price = pd.merge(sentiment_by_time, quarterly_oil, on="Time", how="left")
    sentiment_oil_price = sentiment_oil_price.sort_values("Time")

    r, se = correlation_with_se(sentiment_oil_price["SentimentScore"], sentiment_oil_price["WTI"])
    logger.info(f"Correlation (t): {r:.3f} ± {se:.3f}")

    sentiment_oil_price["WTI_lag1"] = sentiment_oil_price["WTI"].shift(1)
    r, se = correlation_with_se(sentiment_oil_price["SentimentScore"], sentiment_oil_price["WTI_lag1"])
    logger.info(f"Correlation (SentimentScore_t vs WTI_t-1): {r:.3f} ± {se:.3f}")

    sentiment_oil_price["WTI_next1"] = sentiment_oil_price["WTI"].shift(-1)
    r, se = correlation_with_se(sentiment_oil_price["SentimentScore"], sentiment_oil_price["WTI_next1"])
    logger.info(f"Correlation (SentimentScore_t vs WTI_t+1): {r:.3f} ± {se:.3f}")

    # Check deltas
    sentiment_oil_price["Delta_WTI"] = sentiment_oil_price["WTI"].diff()
    sentiment_oil_price["Delta_Sentiment"] = sentiment_oil_price["SentimentScore"].diff()
    sentiment_oil_price["Delta_WTI_next"] = sentiment_oil_price["Delta_WTI"].shift(-1)
    r, se = correlation_with_se(sentiment_oil_price["Delta_Sentiment"], sentiment_oil_price["Delta_WTI_next"])
    logger.info(f"Correlation between ΔSentiment(t) and ΔWTI(t+1): {r:.3f} ± {se:.3f}")
    ## CREATE PLOTS

    ## DELTA PLOT
    plt.figure(figsize=(8, 6))
    plt.scatter(
        sentiment_oil_price["Delta_Sentiment"],
        sentiment_oil_price["Delta_WTI_next"],
        color="#1f77b4",
        alpha=0.8,
        edgecolors="k"
    )

    plt.xlabel("ΔSentiment (this quarter)", fontsize=12)
    plt.ylabel("ΔWTI (next quarter)", fontsize=12)
    plt.title("ΔWTI (t+1) vs. ΔSentiment (t)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("scatter_delta_sentiment_vs_next_wti.png", dpi=300, bbox_inches="tight")

    # Scatter plot for Sentiment vs WTI
    plt.figure(figsize=(8, 6))
    plt.scatter(
        sentiment_oil_price["SentimentScore"],
        sentiment_oil_price["WTI"],
        color="#ff7f0e",
        alpha=0.8,
        edgecolors="k"
    )       
    plt.xlabel("Sentiment Score (-1 to +1)", fontsize=12)
    plt.ylabel("WTI Oil Price ($/bbl)", fontsize=12)
    plt.title("WTI Oil Price vs. Sentiment Score", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("scatter_sentiment_vs_wti.png", dpi=300, bbox_inches="tight")
    
    ## SENTIMENT VS WTI PLOT
    x_labels = sentiment_oil_price["Time"].astype(str)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(x_labels, sentiment_oil_price["SentimentScore"], color="#1f77b4", marker="o", label="Sentiment Score")
    ax1.set_ylabel("Sentiment Score (-1 to +1)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xlabel("Quarter")
    ax1.set_xticks(x_labels)
    ax1.set_xticklabels(x_labels, rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(x_labels, sentiment_oil_price["WTI"], color="#ff7f0e", marker="x", label="WTI Price")
    ax2.set_ylabel("WTI Oil Price ($/bbl)", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    # Title and grid
    plt.title("Sentiment Score vs. WTI Oil Price by Quarter")
    plt.grid(True)
    fig.tight_layout()
    plt.savefig("sentiment_vs_wti.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()