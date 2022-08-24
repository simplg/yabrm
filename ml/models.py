from django.db import models
from django.contrib.auth.models import AbstractUser

# Create your models here.

class User(AbstractUser):
    books_to_read = models.ManyToManyField('Book', blank=True)

class Tag(models.Model):
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class Author(models.Model):
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class Book(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField()
    isbn = models.CharField(max_length=50, null=True)
    isbn13 = models.CharField(max_length=50, null=True)
    publisher = models.CharField(max_length=255, null=True)
    nb_pages = models.IntegerField(null=True)
    image = models.CharField(max_length=255, null=True)
    publication_year = models.IntegerField(null=True)
    language_code = models.CharField(max_length=10, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    tags = models.ManyToManyField(Tag, related_name='books')
    authors = models.ManyToManyField(Author, related_name='books')

    def __str__(self):
        return self.name


class Rating(models.Model):
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    value = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.book.name + " " + str(self.value)
