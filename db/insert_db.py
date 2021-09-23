from __future__ import annotations
import pandas as pd
import numpy as np
import re
from models import Books, BooksTags, Ratings, Tags, ToRead
from db import db

def add_prefix(data: pd.DataFrame, prefix: str, filter: (list | str) =[]):
    if isinstance(filter, str):
        filter = [filter]
    return data.rename(columns={ col: prefix+col if col not in filter else col for col in data.columns})

def insert_data():
    # Chargement des données csv
    ratings = pd.read_csv("../ratings.csv")
    books = pd.read_csv("../books.csv")
    tags = pd.read_csv("../tags.csv")
    book_tags = pd.read_csv("../book_tags.csv")
    to_read = pd.read_csv("../to_read.csv")

    books.isbn13 = books.isbn13.astype("Int64")
    books = add_prefix(books, "book_", "book_id")
    book_tags.drop_duplicates(["goodreads_book_id", "tag_id"], inplace=True)
    ratings = add_prefix(ratings, "rt_", ["book_id", "user_id"])


    with db.begin() as conn:
        Books.insert_df(books, conn)
        print("[1/5] Livres insérés")
        Ratings.insert_df(ratings, conn)
        print("[2/5] Notes insérés")
        Tags.insert_df(tags, conn)
        print("[3/5] Tags insérés")
        ToRead.insert_df(to_read, conn)
        print("[4/5] Liste de lecture inséré")
        BooksTags.insert_df(book_tags, conn)
        print("[5/5] Relation Livres-Tags inséré")
    print("Tout a été inséré !")

if __name__ == '__main__':
    insert_data()