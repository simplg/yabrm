from math import ceil
from sqlalchemy import create_engine
from django.conf import settings
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from ml.models import User


def get_engine():
    return create_engine(f"postgresql://{settings.DATABASES['default']['USER']}:{settings.DATABASES['default']['PASSWORD']}@{settings.DATABASES['default']['HOST']}:5432/{settings.DATABASES['default']['NAME']}")


class TagManager:
    def __init__(self):
        self.__tag_names = []
        self.__relationship = []

    def add_tag_names(self, value: pd.DataFrame):
        self.__tag_names.append(pd.DataFrame(value))

    def add_relationships_by_name(self, relations: pd.DataFrame):
        self.__relationship.append(pd.DataFrame(relations[["name", "book_id"]]))
    
    def insert_tags(self, engine):
        tag_names = pd.concat(self.__tag_names).drop_duplicates(keep="first").reset_index(drop=True)
        tag_names["created_at"] = datetime.now()
        tag_names["updated_at"] = datetime.now()
        tag_names.index += 1
        tag_names.to_sql("ml_tag", engine, if_exists="append", index="id")
        relationships = pd.concat(self.__relationship).drop_duplicates(keep="first")
        tag_names["tag_id"] = tag_names.index
        relationships = relationships.merge(tag_names, on="name")
        for _ in tqdm(chunker_tosql(relationships[["tag_id", "book_id"]], size=10000, name="ml_book_tags", con=engine, if_exists="append", index=False), total=ceil(len(relationships)/10000)):
            pass


class UserManager:
    def __init__(self):
        self.__users = []
        self.__ratings = []
        self.last_user_id = User.objects.latest("id").id

    @property
    def ratings(self):
        return self.__ratings

    def add_users(self, value: pd.DataFrame):
        self.__users.append(value)

    def add_ratings(self, value: pd.DataFrame):
        self.__ratings.append(value)
    
    def prepare_users(self, users : pd.DataFrame):
        print("Generating users infos...")
        users["username"] = users["id"].apply(lambda user_id: f"user_{user_id}")
        users["password"] = "lol"
        users["last_name"] = users["username"]
        users["first_name"] = users["username"]
        users["email"] = users["username"] + "@example.com"
        users["is_superuser"] = False
        users["is_staff"] = False
        users["is_active"] = False
        users["date_joined"] = datetime.now()

    def insert_users(self, engine, low_memory=False):
        print("Inserting users...")
        users = pd.concat(self.__users).drop_duplicates()
        self.prepare_users(users)
        users.to_sql("ml_user", engine, if_exists="append", index=False)
        if low_memory:
            del users
            self.__users = []
        print("Users inserted")

    def insert_ratings(self, engine, low_memory=False):
        print("Inserting ratings...")
        ratings = pd.concat(self.__ratings).drop_duplicates()
        ratings.to_sql("ml_rating", engine, if_exists="append", index=False)
        if low_memory:
            del ratings
            self.__ratings.clear()
        print("Ratings inserted")

    def release(self):
        """Release all resources"""
        self.__users = None
        self.__ratings = None


def chunker_tosql(data: pd.DataFrame, size: int, **kwargs):
    for i in range(0, len(data), size):
        yield data[i:i+size].to_sql(**kwargs)