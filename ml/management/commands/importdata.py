from django.core.management.base import BaseCommand, CommandError
from ._data_transform import import_data
import os

class Command(BaseCommand):
    help = 'Import necessary data from Goodreads'

    def add_arguments(self, parser):
        parser.add_argument('input_dir', type=str)

    def handle(self, *args, **options):
        raw_dir = options["input_dir"]
        if not os.path.isdir(raw_dir):
            raise CommandError('Input directory does not exist')
        needed_files = ["book_id_map.csv", "goodreads_interactions.csv", "goodreads_books.json", "goodreads_book_authors.json"]
        for file in needed_files:
            if not os.path.isfile(os.path.join(raw_dir, file)):
                raise CommandError('Missing file: ' + file)
        self.stdout.write(self.style.SUCCESS('Successfully found all files'))
        import_data(os.path.join(raw_dir, "goodreads_books.json"), os.path.join(raw_dir, "goodreads_interactions.csv"), os.path.join(raw_dir, "book_id_map.csv"), os.path.join(raw_dir, "goodreads_book_authors.json"))
        self.stdout.write(self.style.SUCCESS('Successfully imported data'))