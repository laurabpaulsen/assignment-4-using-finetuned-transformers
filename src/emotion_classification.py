"""
Pipeline for classifying emotions in headlines from fake and real news articles. The following steps are taken:
    - loads data
    - classifies emotions in the headlines
    - saves the data with the predicted emotions

Author: Laura Bock Paulsen (202005791@post.au.dk)
"""
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path


def classify_emotions(df:pd.DataFrame, text_column:str, nlp):
    """
    Classifies emotions in a dataframe of tweets.
    
    Parameters
    ----------

    df : pd.DataFrame
        Dataframe containing the text to perform emotion classification on.
    
    text_column : str
        Name of the column with text
    
    nlp : transformers.pipeline
        Transformers pipeline for emotion classification

    Returns
    -------
    df : pd.DataFrame
        Dataframe containing the text and the predicted emotions
    """
    
    # create a new column for each emotion
    for i, headline in tqdm(enumerate(df[text_column])):
        preds = nlp(headline)

        for pred in preds[0]:
            df.loc[i, pred['label']] = pred['score']

    return df

def main():
    # defining paths
    path = Path(__file__) 
    in_path = path.parents[1] / "data" / "fake_or_real_news.csv"
    out_path = path.parents[1] / "data" / "classified_emotions.csv"

    # loading in the data
    data = pd.read_csv(in_path, usecols = ["title", "label"])

    # defining pipeline for emotion classification
    nlp = pipeline("text-classification", 
                    model="j-hartmann/emotion-english-distilroberta-base", 
                    top_k=None)

    # classifying emotions
    df = classify_emotions(data, 'title', nlp)
    
    # saving data
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()