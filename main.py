
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)



df = pd.read_csv("amazon_review/amazon_review.csv")
df.head()
df.shape



df["overall"].mean()


df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date = df["reviewTime"].max()

df["current_diff"] = (current_date - df["reviewTime"]).dt.days

df["current_diff"].quantile([0.1,0.25,0.50])

df.loc[df["current_diff"] <= 166, "overall"].mean() #4.68986083499006

df.loc[(df["current_diff"] > 166) & (df["current_diff"] <= 280), "overall"].mean() #4.699863574351978

df.loc[(df["current_diff"] > 280) & (df["current_diff"] <= 430), "overall"].mean() #4.636140637775961

df.loc[(df["current_diff"] > 430), "overall"].mean() #4.508957654723127


def time_based_weighted_average(dataframe, w1=30, w2=28, w3=24, w4=18):
    return dataframe.loc[df["current_diff"] <= 166, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["current_diff"] > 166) & (dataframe["current_diff"] <= 280), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["current_diff"] > 280) & (dataframe["current_diff"] <= 430), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["current_diff"] > 430), "overall"].mean() * w4 / 100

time_based_weighted_average(df)




df["helpful_no"] = (df.total_vote - df.helpful_yes) #yararsız bulunan oy sayısı
df["helpful_no"].head()


def score_pos_neg_diff(up,down):
    return up - down


def score_average_rating(helpful_yes, helpful_no):
    if helpful_yes + helpful_no == 0:
        return 0
    return helpful_yes / (helpful_yes + helpful_no)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df["score_pos_neg_diff"].sort_values(ascending=False).head(20)

df["score_average_rating"].sort_values(ascending=False).head(20)

df["wilson_lower_bound"].sort_values(ascending=False).head(20)

df.sort_values("wilson_lower_bound", ascending=False).head(20)
