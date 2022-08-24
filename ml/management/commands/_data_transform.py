from datetime import datetime
import itertools
from math import ceil
import pandas as pd
import re

from tqdm import tqdm
import django
django.setup()
from multiprocessing import Pool, cpu_count
from ml.helpers import get_engine, TagManager, UserManager, chunker_tosql
from ml.models import User


def import_data(books_path: str, interactions_path: str, book_id_map_path: str, authors_path: str):
    tag_manager = TagManager()
    user_manager = UserManager()
    print ("Last inserted user id:", user_manager.last_user_id)
    print("Inserting authors...")
    insert_authors(authors_path)
    print("inserting books...")
    book_id_list = insert_books(tag_manager, books_path)
    print("Inserting tags...")
    insert_tags(tag_manager)
    print("Inserting users and ratings...")
    read_ratings_csv(user_manager, interactions_path, book_id_map_path, book_id_list)
    insert_users_and_ratings(user_manager)
    user_manager.release()


def extract_book_ratings(data: pd.DataFrame, book_ids_replace_map: dict):
    data["book_id"] = data["book_id"].map(book_ids_replace_map)
    data.dropna(inplace=True)
    data["book_id"] = data["book_id"].astype(int)
    data["user_id"] = data["user_id"].astype(int)
    data["rating"] = data["rating"].astype(int)
    data.rename(columns={"rating": "value"}, inplace=True)
    return data


def filter_tags(row: str):
    exclude_tags = ["to-read", "favorites", "owned", "books-i-own", "currently-reading", "library", "owned-books", "to-buy", "kindle", "default", "ebook", "my-books", "audiobook", "ebooks", "wish-list", "my-library", "audiobooks", "i-own", "favourites", "audio", "own-it", "e-book", "books", "audible", "audio-books", "abandoned", "re-read", "have", "audio-book", "borrowed", "english", "did-not-finish", "favorite", "maybe", "shelfari-favorites", "ya", "all-time-favorites", "favorite-books", "dnf", "finished", "paperback", "reviewed", "unfinished", "home-library", "library-books", "calibre", "didn-t-finish", "to-read-fiction", "nook", "library-book", "favorite-authors", "want-to-read", "tbr", "unread", "recommended", "bookshelf", "books-i-have", "own-to-read", "kindle-books", "must-read", "need-to-buy", "read-in-english", "reread", "on-my-shelf", "ya-books", "my-favorites", "listened-to", "on-hold", "favorite-series", "on-kindle", "to-read-non-fiction", "shelfari-wishlist", "ya-fantasy", "book-club-books", "personal-library", "my-bookshelf", "book-group", "other", "faves", "couldn-t-finish", "gave-up-on", "mine", "childhood-favorites", "want", "ya-lit", "to-be-read", "to-read-nonfiction", "hardcover", "read-for-school", "favourite", "non-fiction-to-read", "favs", "never-finished", "in-my-library", "childhood-reads", "collection", "books-to-buy", "to-get", "on-my-bookshelf", "book-club-reads", "book-boyfriends", "read-more-than-once", "bought", "to-read-own", "loved", "owned-to-read", "fiction-to-read", "gave-up", "want-to-buy", "to-read-classics", "purchased", "books-owned", "read-aloud", "ya-romance", "scanned", "netgalley", "my-ebooks", "meh", "i-own-it", "on-my-kindle", "home", "signed", "book-boyfriend", "my-childhood", "to-reread", "to-read-fantasy", "read-as-a-kid", "chapter-books", "already-read", "favorite-author", "to-read-owned", "ya-paranormal", "not-read", "read-alouds", "stopped-reading", "summer-reading", "sub", "book", "my-collection", "series-to-read", "could-not-finish", "loved-it", "ya-contemporary", "not-interested", "partially-read", "read-as-a-child", "listened", "want-to-own", "beach-reads", "childhood-favourites", "done", "re-reads", "read-fiction", "books-from-my-childhood", "tbr-pile", "next", "series-to-finish", "have-read", "fantasy-read", "read-fantasy", "read-again", "and", "summer-reads", "in-english", "paper", "fae", "fantasy-to-read", "work", "need", "buy", "ibooks", "own-kindle", "self", "new", "pdf", "biblioteca", "in-translation", "amazon", "all-time-favourites", "own-a-copy", "hardback", "epub", "not-finished", "collections", "to-purchase", "donated", "paused", "a", "recommendations", "completed", "translations", "unfinished-series", "dropped"]
    exclude_regexp = [re.compile(r"(?:books-)*read(?:-in)*-[0-9]+"), re.compile(r"^[0-9]+"), re.compile(r"(?:challenge|rory|read|favorite|shelf|book|own|childhood|gilmore|favo[u]*rite|kindle|buy|finish)")]
    return exclude_tags.count(row) == 0 | sum(map(lambda reg: reg.search(row) != None, exclude_regexp)) == 0


def transform_books(data: pd.DataFrame):
    data["book_id"] = data["book_id"].astype(int)
    data["publication_year"] = pd.to_numeric(data["publication_year"], errors="coerce", downcast="integer")
    data["num_pages"] = pd.to_numeric(data["num_pages"], errors="coerce", downcast="integer")
    data.rename(columns={"book_id": "id", "num_pages": "nb_pages", "image_url": "image", "title": "name"}, inplace=True)
    return data


def get_book_id_replacer(book_id_map_path: str):
    book_id_map = pd.read_csv(book_id_map_path)
    return { id: id_csv for id_csv, id in zip(book_id_map["book_id_csv"], book_id_map["book_id"]) }


def transform_authors(authors: pd.DataFrame):
    authors["author_id"] = authors["author_id"].astype(int)
    authors["created_at"] = datetime.now()
    authors["updated_at"] = datetime.now()
    authors.rename(columns={"author_id": "id"}, inplace=True)
    authors = authors[["id", "name", "created_at", "updated_at"]]
    return authors


def insert_ratings(ratings: pd.DataFrame):
    ratings.drop_duplicates(inplace=True)
    ratings["created_at"] = datetime.now()
    ratings["updated_at"] = datetime.now()
    engine = get_engine()
    with engine.connect() as conn:
        for _ in tqdm(chunker_tosql(ratings, 10000, name="ml_rating", con=conn, if_exists="append", index=False), total=ceil(len(ratings)/10000)):
            pass


def read_ratings_csv(user_manager: UserManager, interactions_path: str, book_id_map_path: str, book_id_list: list[int]):
    with pd.read_csv(interactions_path, chunksize=1000000) as ratings_chunk:
        print("Estimating number of ratings to import...")
        with Pool(processes=cpu_count()) as pool:
            with tqdm(total=ceil(sum(1 for _ in open(interactions_path))/1000000)) as pbar:
                for (ratings, users) in pool.imap(extract_and_import_ratings, zip(ratings_chunk, itertools.repeat(book_id_map_path), itertools.repeat(user_manager.last_user_id))):
                    user_manager.add_users(users)
                    user_manager.add_ratings(ratings[ratings["book_id"].isin(book_id_list)])
                    pbar.update(1)


def insert_users_and_ratings(user_manager: UserManager):
    engine = get_engine()
    with engine.connect() as conn:
        with conn.begin():
            user_manager.insert_users(conn, True)
    with Pool(processes=cpu_count()) as pool:
        with tqdm(total=len(user_manager.ratings)) as pbar:
            for _ in pool.imap(insert_ratings, user_manager.ratings):
                pbar.update(1)

def insert_authors(authors_path: str):
    engine = get_engine()
    with engine.connect() as conn:
        authors = transform_authors(pd.read_json(authors_path, lines=True))
        authors.to_sql("ml_author", conn, if_exists="append", index=False)


def insert_tags(tag_manager: TagManager):
    engine = get_engine()
    with engine.connect() as conn:
        tag_manager.insert_tags(conn)


def insert_books(tag_manager: TagManager, books_path: str) -> list[int]:
    books_iter = pd.read_json(books_path, lines=True, chunksize=1000)
    print("Estimating number of books to import...")
    book_id_list = []
    with Pool(processes=cpu_count()) as pool:
        with tqdm(total=ceil(sum(1 for _ in open(books_path))/1000)) as pbar:
            for (tags, relations, book_ids) in pool.imap_unordered(import_books, books_iter):                    
                tag_manager.add_tag_names(tags)
                tag_manager.add_relationships_by_name(relations)
                book_id_list.extend(book_ids.to_list())
                pbar.update(1)
    return book_id_list



def import_books(books: pd.DataFrame):
    engine = get_engine()
    with engine.connect() as conn:
        with conn.begin():
            books = transform_books(books)
            prepare_books(books)
            books_table = books[["id", "name", "description", "isbn", "isbn13", "publisher", "language_code", "publication_year", "nb_pages", "image", "created_at", "updated_at"]]
            books_table.to_sql("ml_book", conn, if_exists="append", index=False)
            tags = prepare_tags(books)
            authors = prepare_authors(books)
            authors.to_sql("ml_book_authors", conn, if_exists="append", index=False)
            return tags["name"].drop_duplicates().copy(), transform_tag_relationship(tags[["book_id", "name"]].copy()), books["id"]

def prepare_tags(books: pd.DataFrame):
    tags = books[["popular_shelves", "id"]]
    tags = tags.explode("popular_shelves").copy().dropna().reset_index(drop=True)
    tags.rename(columns={"id": "book_id"}, inplace=True)
    tags = pd.concat([tags["book_id"], pd.DataFrame(tags["popular_shelves"].to_list())], axis=1)
    tags["count"] = tags["count"].astype(int)
    tags = tags[(tags["count"] > 10) & tags["name"].apply(filter_tags)]
    return tags


def prepare_books(books: pd.DataFrame):
    books["created_at"] = datetime.now()
    books["updated_at"] = datetime.now()


def prepare_authors(books: pd.DataFrame):
    authors = books[["authors", "id"]]
    authors = authors.explode("authors").copy().dropna().reset_index(drop=True)
    authors.rename(columns={"id": "book_id"}, inplace=True)
    authors = pd.concat([authors["book_id"], pd.DataFrame(authors["authors"].to_list())["author_id"]], axis=1)
    authors.drop_duplicates(inplace=True)
    return authors


def transform_ratings(ratings: pd.DataFrame, last_user_id: int):
    value_copy = ratings[["user_id", "book_id", "value"]].copy()
    return value_copy

def transform_users(ratings: pd.DataFrame, last_user_id: int):
    value_copy = pd.DataFrame(ratings["user_id"].drop_duplicates())
    value_copy.rename(columns={"user_id": "id"}, inplace=True)
    return value_copy

def extract_and_import_ratings(args: tuple[pd.DataFrame, str]):
    ratings, book_id_map_path, last_user_id = args
    book_id_replacer = get_book_id_replacer(book_id_map_path)
    ratings = extract_book_ratings(ratings, book_id_replacer)
    ratings["user_id"] += last_user_id + 1
    return transform_ratings(ratings, last_user_id), transform_users(ratings, last_user_id)

def transform_tag_relationship(relations: pd.DataFrame):
    return relations.drop_duplicates()