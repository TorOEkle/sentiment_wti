import pandas as pd
import matplotlib.pyplot as plt
import logging
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

    correlation = sentiment_oil_price["SentimentScore"].corr(sentiment_oil_price["WTI"])
    logger.info(f"Correlation (Pearson): {correlation:.3f}")

    sentiment_oil_price["WTI_lag1"] = sentiment_oil_price["WTI"].shift(1)
    lagged_corr = sentiment_oil_price["SentimentScore"].corr(sentiment_oil_price["WTI_lag1"])
    logger.info(f"Correlation with WTI (lagged 1 quarter): {lagged_corr:.3f}")

    ## CREATE PLOT

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