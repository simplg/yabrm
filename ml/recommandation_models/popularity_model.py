import pandas as pd
import numpy as np
from itertools import chain

class PopularityBasedModel():
    def __init__(self, percentile: int=70, rating_column: str="avg_rating", vote_count_column: str="vote_count", book_id_column: str="book_id"):
        self.sorted_books_id: list[int] = []
    
    def fit(self, ratings: pd.DataFrame):
        C = np.mean(ratings['avg_rating'])
        m = np.percentile(ratings['vote_count'], 70)
        ratings_filtered = ratings[ratings['vote_count'] >= m]
        R = ratings_filtered['avg_rating']
        v = ratings_filtered['vote_count']
        ratings_filtered['weighted_rating'] = self.weighted_rating(v,m,R,C)
        self.sorted_books_id = ratings_filtered.sort_values('weighted_rating', ascending=False)['book_id'].tolist()
        

    def predict(self, slices: list[tuple[int, int]]=[(0,5), (45,50)]) -> list[int]:
        # On récupère les livres ordonnées selon le nombre de vote
        sorted_books = self.sorted_books_id
        return chain.from_iterable(sorted_books[start:end] for (start, end) in slices)
    
    def weighted_rating(self,v,m,R,C):
        '''
        Calculate the weighted rating
        
        Args:
        v -> average rating for each item (float)
        m -> minimum votes required to be classified as popular (float)
        R -> average rating for the item (pd.Series)
        C -> average rating for the whole dataset (pd.Series)
        
        Returns:
        pd.Series
        '''
        return ( (v / (v + m)) * R) + ( (m / (v + m)) * C )