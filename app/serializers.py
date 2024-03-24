from django.contrib.auth.models import User
from rest_framework import serializers

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'password', 'email']
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        return user



from rest_framework import serializers
from django.core.files.uploadedfile import UploadedFile
from .models import *

# class PdfSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Pdf
#         fields = ['pdf']
#
#     def to_internal_value(self, data):
#         # Check if the incoming data is an uploaded file
#         if isinstance(data, UploadedFile):
#             # If it's an uploaded file, return a dictionary with the 'pdf' key
#             return {'pdf': data}
#         else:
#             # If it's not an uploaded file, use the default behavior
#             return super().to_internal_value(data)

from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Pdf

class PdfSerializer(serializers.ModelSerializer):
    class Meta:
        model = Pdf
        fields = ['pdf']

# Assuming you want to upload multiple PDFs with a username
class UploadPdfSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    pdfs = PdfSerializer(many=True, required=True)

    def create(self, validated_data):
        username = validated_data.get('username')
        pdfs_data = validated_data.get('pdfs')

        # Fetch the user based on the username
        user, _ = User.objects.get_or_create(username=username)

        # Create Pdf instances for each uploaded file
        pdfs = [Pdf(user=user, pdf=pdf_data['pdf']) for pdf_data in pdfs_data]
        Pdf.objects.bulk_create(pdfs)

        return {'username': username, 'pdfs': pdfs}

