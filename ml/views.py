from genericpath import isfile
import os
from typing import Any
from django.http import JsonResponse
from django.shortcuts import render
from django.views.generic.list import ListView
from django.views.generic.detail import DetailView
import pickle
import tensorflow as tf
import pandas as pd
from ml.helpers import get_engine

from ml.models import Book
from ml.recommandation_models.popularity_model import PopularityBasedModel

# Create your views here.

popularity_model = pickle.load(open('data/output/pop_model.pkl', 'rb'))
if os.path.isdir('data/output/bst_model') and len(os.listdir('data/output/bst_model')) > 0:
    collab_model =  tf.keras.models.load_model('data/output/bst_model')

class BookListView(ListView):
    model = Book
    context_object_name = 'books'
    paginate_by = 12
    
    def get_queryset(self):
        query = self.request.GET.get('q')
        if query:
            return Book.objects.filter(name__icontains=query).order_by('name')
        return Book.objects.order_by('name').all()
    
    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        context['page_obj'].additional_args = f"q={self.request.GET.get('q', '')}"
        return context

class BookDetailView(DetailView):
    model = Book
    context_object_name = 'book'
    template_name = 'ml/book_detail.html'
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['ratings'] = self.object.rating_set.all()
        return context

def index(request):
    last_books = Book.objects.all().order_by('-created_at')[:10]
    return render(request, 'ml/index.html', {'last_books': last_books})

def popularity():
    book_ids = popularity_model.predict()
    books = Book.objects.filter(id__in=book_ids)
    return JsonResponse({'books': [book.to_dict() for book in books]})

def collaborative(request):
    books_liked = request.POST.getlist('books_liked')
    ratings = request.POST.getlist('ratings')
    book_list = { book_id: rating for book_id, rating in zip(books_liked, ratings) }
    book_ids = collab_model.predict(book_list)
    books = Book.objects.filter(id__in=book_ids)
    return JsonResponse({'books': [book.to_dict() for book in books]})