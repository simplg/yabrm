# Generated by Django 4.0.5 on 2022-06-30 10:20

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ml', '0006_remove_author_goodreads_id'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='book',
            name='goodreads_id',
        ),
    ]
