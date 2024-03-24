# Generated by Django 4.2.11 on 2024-03-07 10:44

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Pdf',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pdf', models.FileField(upload_to='documents/pdfs/')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='pdfs', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
