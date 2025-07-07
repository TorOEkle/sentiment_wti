# Sentiment in energy survey from Dallas Fed
I was curious if the sentiments from comments in the Dallas Fed Energy Survey could be correlated with WTI oil prices. The atteched scripts scrape the survey comments from 2017 to 2025, classify the sentiment, create a plot and print correlation. It might not be suprise, but the SentimentScore has a higher absolute value in correlation for previous period of WTI prices. More suprisingly for me was it had a negative sign.  $corr(SentimentScore, wti_{t-1}) = -0.328  ± 0.160$ than $corr(SentimentScore, wti_{t+1}) = -0.152 ± 0.175$ One reason might be that the industry expect worse times ahead when prices are high and vice versa.

**PS:** The sentiment analysis in this project is not very elegant, but it serves as a basic example of how to analyze sentiment in relation to WTI oil prices. I have used the model `cardiffnlp/twitter-roberta-base-sentiment` from Hugging Face for sentiment classification. Tested also with `ProsusAI/finbert`, but it did not perform as well in terms of correlation with WTI prices.




Oil price is taken from [FRED](https://fred.stlouisfed.org/series/DCOILWTICO).

### Virtual Environment Setup
This project uses [uv](https://github.com/astral-sh/uv) for environment and dependency management.

1. **Initialize the project with uv:**
    ```bash
    uv init
    ```

2. **Install required packages:**
    ```bash
    uv pip install \
      pandas \
      matplotlib \
      textblob \
      beautifulsoup4 \
      requests \
      spacy \
      transformers \
      torch
    ```
Yes I use both uv and pip, because uv does not support installing spaCy models directly.

3. **Download the spaCy English model:**
    ```bash
    uv venv exec python -m spacy download en_core_web_sm
    ```

4. **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

5. **Run your scripts:**
Run `scrape_classify.py` first to scrape and classify the data, then run `sentiment_wti.py` to create plot of sentiment and WTI prices.


## Plot
The resulting plot frome my analysis is as follows:
![Sentiment vs WTI Oil Prices](sentiment_vs_wti.png)
