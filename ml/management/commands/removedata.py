from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from ml.models import User

class Command(BaseCommand):
    help = 'Import necessary data from Goodreads'

    def handle(self, *args, **options):
        if (input("Are you sure you want to remove all data? (y/n) ") != "y"):
            return
        with transaction.atomic():
            User.objects.filter(goodreads_id__isnull=False).delete()