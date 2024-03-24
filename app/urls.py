from django.urls import path
from .views import *

urlpatterns = [
    path('signup/', UserCreate.as_view(), name='user_signup'),
    path('login/', LoginView.as_view(), name='user_signup'),
    path('profile/', UserProfileView.as_view(), name='user_profile'),
    path('upload-pdfs/', UploadPdfsView.as_view(), name='upload-pdfs'),
    path('chat-pdf/', ChatPdfView.as_view(), name='chat-pdf'),
    path('logout/', LogoutView.as_view(), name='logout'),

  # Add more paths for login and other endpoints
]
