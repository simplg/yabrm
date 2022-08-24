from django.core.management.base import BaseCommand, CommandError
from ._recommandation_model import train_popularity_model, prepare_csv_bst_model, train_bst_model

class Command(BaseCommand):
    help = "Train the recommandation model"

    def handle(self, *args, **options):
        # train_popularity_model()
        self.stdout.write(self.style.SUCCESS("Successfully trained the popularity model"))
        prepare_csv_bst_model()
        self.stdout.write(self.style.SUCCESS("Successfully prepared the csv file for the bst model"))
        train_bst_model()
        self.stdout.write(self.style.SUCCESS("Successfully trained the bst model"))

        self.stdout.write(self.style.SUCCESS('Successfully trained the recommandation model'))
        return 0