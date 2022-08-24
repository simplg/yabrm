from django.urls import path

from ml.views import BookListView, BookDetailView, index, popularity

app_name = 'ml'

urlpatterns = [
    path('books', BookListView.as_view(), name='book_list'),
    path('books/<int:pk>', BookDetailView.as_view(), name='book_detail'),
    path('popularity', popularity, name='popularity'),
    path('', index, name='index'),
]