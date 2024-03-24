from django.db import models
from django.conf import settings
import datetime


# class Document(models.Model):
#   user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, default=1)
#   description = models.TextField()
#   date = models.DateField(default=datetime.date.today)
#
# class Pdf(models.Model):
#   document = models.ForeignKey(Document, related_name='pdfs', on_delete=models.CASCADE)
#   pdf = models.FileField(upload_to='documents/pdfs/')

#
# class Image(models.Model):
#   document = models.ForeignKey(Document, related_name='images', on_delete=models.CASCADE)
#   image = models.FileField(upload_to='documents/images/')




from django.db import models
from django.conf import settings

class Pdf(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='pdfs')
    pdf = models.FileField(upload_to='documents/pdfs/')

    def __str__(self):
        return f"{self.user.username} - {self.pdf.name}"
