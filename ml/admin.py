from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from ml.models import Author, Book, Tag, User

# Register your models here.
admin.site.register(User, UserAdmin)
admin.site.register(Book)
admin.site.register(Tag)
admin.site.register(Author)