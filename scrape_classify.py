import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
from functions import THEME_KEYWORDS, get_sentiment, detect_themes, detect_segment
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    BASE_URL = "https://www.dallasfed.org/research/surveys/des"

    all_data = []

    for year in range(2017, 2026):  
        for quarter in range(1, 5):  
            report_id = f"{year % 100:02d}{quarter:02d}"
            url = f"{BASE_URL}/{year}/{report_id}"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    logger.warning(f"Skipped: {url} (status {response.status_code})")
                    continue

                soup = BeautifulSoup(response.text, "html.parser")

                # Extract comments
                comment_section = soup.find("div", {"id": "tab-comments"})
                comment_items = comment_section.find_all("li") if comment_section else []

                for item in comment_items:
                    text = item.get_text(strip=True)
                    if text:
                        all_data.append({
                            "Comment": text,
                            "Source": "Comment",
                            "Time": f"{year}Q{quarter}"
                        })

                # Extract questions
                question_section = soup.find("div", {"id": "tab-questions"})
                question_items = question_section.find_all("li") if question_section else []

                for item in question_items:
                    text = item.get_text(strip=True)
                    if text:
                        all_data.append({
                            "Comment": text,
                            "Source": "Question",
                            "Time": f"{year}Q{quarter}"
                        })

                print(f"Collected {len(comment_items) + len(question_items)} items from {year}Q{quarter}")

            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")

    # Convert to DataFrame and tag
    df = pd.DataFrame(all_data)
    # Analysis
    df["Sentiment"] = df["Comment"].apply(get_sentiment)
    df["Theme(s)"] = df["Comment"].apply(lambda x: ", ".join(detect_themes(x)))
    df["Segment"] = df["Comment"].apply(detect_segment)

    df.to_csv("dallas_fed_NLP.csv", index=False)

    # Generate sentiment summary
    df_exploded = df.assign(Theme=df["Theme(s)"].str.split(",")).explode("Theme")
    summary = (
        df_exploded.groupby(["Theme", "Sentiment","Time"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    print(summary)
    logger.info(df[["Comment", "Sentiment","Time"]].head(10))

if __name__ == "__main__":
    main()
