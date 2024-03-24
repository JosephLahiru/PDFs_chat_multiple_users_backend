# from django.contrib.auth.models import User
# from rest_framework import generics
# from .serializers import UserSerializer
#
# class UserCreate(generics.CreateAPIView):
#     queryset = User.objects.all()
#     serializer_class = UserSerializer
#     print(serializer_class)

from django.contrib.auth.models import User
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import UserSerializer

from rest_framework.permissions import AllowAny
from rest_framework.permissions import IsAuthenticated
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

# @method_decorator(csrf_exempt, name='dispatch')
class UserCreate(APIView):
    permission_classes = [AllowAny]
    def post(self, request, *args, **kwargs):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            print("Incoming data:", request.data)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            print("Error:", serializer.errors)
            print("Incoming data:", request.data)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

from django.contrib.auth import authenticate, login
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.contrib.auth.models import User
from .serializers import UserSerializer  # Ensure your serializer is correctly imported

import logging

logger = logging.getLogger(__name__)

# class LoginView(APIView):
#     permission_classes = [AllowAny]
#
#     def post(self, request, *args, **kwargs):
#         logger.debug("Request headers: %s", request.headers)
#         logger.debug("Request cookies: %s", request.COOKIES)
#         # Existing authentication logic
#
#         username = request.data.get('username')
#         password = request.data.get('password')
#         user = authenticate(request, username=username, password=password)
#         if user is not None:
#             login(request, user)
#             print('login')
#             return Response({"message": "User logged in successfully"}, status=status.HTTP_200_OK)
#         else:
#             return Response({"error": "Invalid username or password"}, status=status.HTTP_400_BAD_REQUEST)

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.contrib.auth import authenticate, login
import logging

logger = logging.getLogger(__name__)

class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        logger.debug("Request headers: %s", request.headers)
        logger.debug("Request cookies: %s", request.COOKIES)

        username = request.data.get('username')
        password = request.data.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            # Print the staff status of the user
            print(f"{username} staff status: {user.is_staff}")
            return Response({"message": "User logged in successfully", "is_staff": user.is_staff}, status=status.HTTP_200_OK)
        else:
            return Response({"error": "Invalid username or password"}, status=status.HTTP_400_BAD_REQUEST)


# Import logout
from django.contrib.auth import logout
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class LogoutView(APIView):
    permission_classes = [IsAuthenticated]  # Only authenticated users should be able to logout

    def post(self, request, *args, **kwargs):
        logout(request)
        return Response({"message": "User logged out successfully"}, status=status.HTTP_200_OK)

from django.contrib.auth.models import User
from rest_framework import permissions, generics
from rest_framework.response import Response
from .serializers import UserSerializer  # Assuming you have a UserSerializer

class UserProfileView(generics.RetrieveAPIView):
    permission_classes = [permissions.IsAuthenticated]  # Ensure the user is logged in
    print(permission_classes)
    serializer_class = UserSerializer

    def get_object(self):
        # Return the user making the request
        return self.request.user

# views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth.models import User
from .models import Pdf
from .serializers import PdfSerializer
#
# class UploadPdfsView(APIView):
#     permission_classes = [AllowAny]
#
#     def post(self, request, format=None):
#         username = request.data.get('username')
#         pdf_files = request.FILES.getlist('pdfs')
#         print('pdf_files',pdf_files)
#         # loader = PyPDFLoader(PDF_PATH)
#         # pdf_content = loader.load()
#         # print(pdf_content)
#
#         user = User.objects.filter(username=username).first()
#         if not user:
#             return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
#
#         for pdf_file in pdf_files:
#             Pdf.objects.create(user=user, pdf=pdf_file)
#
#         return Response({"message": "PDFs uploaded successfully"}, status=status.HTTP_201_CREATED)

import PyPDF2
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader
import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from langchain.text_splitter import CharacterTextSplitter,TokenTextSplitter
# Check input length of the model to adjust for not loosing info
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 30

from langchain_community.vectorstores import Chroma
from deep_translator import GoogleTranslator

HUGGINGFACEHUB_API_TOKEN = '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'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

embed_model = HuggingFaceInferenceAPIEmbeddings( api_key=HUGGINGFACEHUB_API_TOKEN, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" )


class UploadPdfsView(APIView):
    permission_classes = [AllowAny]

    def post(self, request, format=None):
        username = request.data.get('username')
        pdf_files = request.FILES.getlist('pdfs')
        print('pdf_files',pdf_files)
        # loader = PyPDFLoader(PDF_PATH)
        # pdf_content = loader.load()
        # print(pdf_content)
        print(username)
        user = User.objects.filter(username=username).first()
        if not user:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        all_pdfs = []
        for pdf_file in pdf_files:
          # Generate a temporary file path
          temp_file_path = default_storage.save("temp_files/" + pdf_file.name, ContentFile(pdf_file.read()))

          try:
            # The file is now saved temporarily on disk. Pass the path to PyPDFLoader.
            loader = PyPDFLoader(temp_file_path)
            pdf_content = loader.load()
            all_pdfs+= pdf_content
            # print(pdf_content)
          finally:
            # Clean up: remove the temporary file
            os.remove(temp_file_path)

          Pdf.objects.create(user=user, pdf=pdf_file)

        if len(all_pdfs)>0 :
          persist_dir = "./vector_databases/" + username + "_chroma_db"
          text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separator="\n", )
          docs = text_splitter.split_documents(all_pdfs)
          for i, doc in enumerate(docs):
            print('Translating', i, '/', len(docs))
            # print(len(doc.page_content))
            to_translate = doc.page_content

            translated = GoogleTranslator(source='auto', target='en').translate(to_translate)
            # print(translated)
            doc.page_content = translated
          if os.path.exists(persist_dir):
            print('persist_dir exists :',persist_dir)
            vectordb = Chroma(persist_directory=persist_dir, embedding_function=embed_model)
            print(vectordb._collection.count())

            vectordb.add_documents(docs)
            vectordb.persist()
            print(vectordb._collection.count())


          else :
            print('persist_dir doesnt exist :', persist_dir)
            vectordb = Chroma.from_documents(docs, embed_model, persist_directory=persist_dir)
            print(vectordb._collection.count())

        return Response({"message": "PDFs uploaded successfully"}, status=status.HTTP_201_CREATED)

from django.contrib.auth.models import User
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

# class ChatPdfView(APIView):
#     permission_classes = [IsAuthenticated]
#
#     def get(self, request, *args, **kwargs):
#         username = request.user.username
#         print(username)  # This prints the username to the console
#         return Response({"username": username})

from rest_framework import generics, permissions
from rest_framework.response import Response


# class ChatPdfView(generics.RetrieveAPIView):
#   permission_classes = [permissions.IsAuthenticated]
#
#   def get(self, request, *args, **kwargs):
#     # Retrieve the username of the request user
#     username = request.user.username
#
#     # Print the username to the console
#     print(username)
#
#     # Send the username to the Angular component
#     return Response({"username": username})

from rest_framework import generics, permissions
from rest_framework.response import Response
# GET USERNAME
# class ChatPdfView(generics.RetrieveAPIView):
#     permission_classes = [permissions.IsAuthenticated]
#     username = ''  # Class attribute to hold the username
#
#     def dispatch(self, request, *args, **kwargs):
#         # Set the username before the request is handled by the specific method
#         self.username = request.user.username
#         print(self.username)  # Optional: print the username here for debugging
#         return super().dispatch(request, *args, **kwargs)
#
#     def get(self, request, *args, **kwargs):
#         # Use self.username as needed here
#         # Since self.username is already set in dispatch, you can directly use it
#         return Response({"username": self.username})

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

HUGGINGFACEHUB_API_TOKEN = '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'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


embed_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGINGFACEHUB_API_TOKEN, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# persist_dir = "./vector_databases/" + username + "_chroma_db"
#
# vectordb = Chroma(persist_directory=persist_dir, embedding_function=embed_model)
# # print(vectordb)
# print(vectordb._collection.count())
#

from rest_framework import generics, permissions
from rest_framework.response import Response
from langchain_community.vectorstores import Chroma
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

# @method_decorator(csrf_exempt, name='dispatch')
# class ChatPdfView(generics.RetrieveAPIView):
#     # permission_classes = [permissions.IsAuthenticated]
#     permission_classes = [AllowAny]
#     username = ''  # This will store the username
#
#     def dispatch(self, request, *args, **kwargs):
#         # Set the username before the request is specifically handled
#         self.username = request.user.username
#         return super().dispatch(request, *args, **kwargs)
#
#     def get(self, request, *args, **kwargs):
#         # Initialize vector database and get the vectordb instance
#         vectordb = self.initialize_vector_db()
#
#         # Now print the count of items in the vector database within the get method
#         print(vectordb._collection.count())
#
#         return Response({"username": self.username})
#
#     def initialize_vector_db(self):
#         persist_dir = "./vector_databases/" + self.username + "_chroma_db"
#         vectordb = Chroma(persist_directory=persist_dir, embedding_function=embed_model)
#         return vectordb  # Return the vectordb instance for further use
#
#     def post(self, request, *args, **kwargs):
#       message = request.data.get('message')
#       print(message)  # Log the received message
#       chat_response = 'Hello'  # Your fixed response
#       return Response({"chat_response": chat_response})

# class ChatPdfView(generics.RetrieveAPIView):
#   permission_classes = [permissions.IsAuthenticated]
#   # permission_classes = [AllowAny]
#
#   def post(self, request, *args, **kwargs):
#     message = request.data.get('message')
#     print(message)  # Log the received message
#     chat_response = 'Hello'  # Your fixed response
#     return Response({"chat_response": chat_response})

# class ChatPdfView(generics.RetrieveAPIView):
#     permission_classes = [permissions.IsAuthenticated]
#
#     def post(self, request, *args, **kwargs):
#       print(request.data)
#       print(f"Authenticated username: {request.user.username}")
#
#       message = request.data.get('message')
#       print(message)  # Log the received message
#       chat_response = 'Hello'  # Your fixed response
#       return Response({"chat_response": chat_response})
import secrets
from rest_framework import generics, permissions
from rest_framework.response import Response


# class ChatPdfView(generics.RetrieveAPIView):
#     permission_classes = [permissions.IsAuthenticated]
#     encryption_key_generated = False
#     encryption_key = ""
#
#     def get(self, request, *args, **kwargs):
#         return Response({"username": request.user.username})
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Check if the encryption key has already been generated
#         if not ChatPdfView.encryption_key_generated:
#             # Assuming there's a way to access the username at this point,
#             # which might need adjustment based on your actual user handling logic.
#             # This is a placeholder for where and how you'd get the username.
#             username = "placeholder_for_username"
#             ChatPdfView.encryption_key = generate_encryption_key(username)
#             ChatPdfView.encryption_key_generated = True
#
#     def post(self, request, *args, **kwargs):
#         print(request.data)
#         print(f"Authenticated username: {request.user.username}")
#         # Since the encryption key is generated in the __init__, just use it
#         print(f"Encryption Key: {self.encryption_key}")
#
#         message = request.data.get('message')
#         print(message)  # Log the received message
#         chat_response = 'Hello'  # Your fixed response
#         return Response({"chat_response": chat_response})

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

# class ChatPdfView(generics.RetrieveAPIView):
#     permission_classes = [IsAuthenticated]
#
#     def get(self, request, *args, **kwargs):
#         return Response({"username": request.user.username})
#     def post(self, request, *args, **kwargs):
#         print(request.data)
#         print(f"Authenticated username: {request.user.username}")
#
#         message = request.data.get('message')
#         print(message)  # Log the received message
#         chat_response = 'Hello'  # Your fixed response
#         return Response({"chat_response": chat_response})

from rest_framework import generics, permissions
from rest_framework.response import Response
import secrets

from rest_framework import generics, permissions
from rest_framework.response import Response
import secrets


# class ChatPdfView(generics.RetrieveAPIView):
#   permission_classes = [permissions.IsAuthenticated]
#   # Initialization right here to ensure it's easily visible for demonstration
#   encryption_key = ''
#
#   @classmethod
#   def generate_encryption_key(cls, username):
#     # Directly generate and update the class variable
#     cls.encryption_key = f"{username}_{secrets.token_hex(8)}"
#
#   def get(self, request, *args, **kwargs):
#     # If the key hasn't been generated yet, generate it using the username
#     if not self.encryption_key:
#       self.generate_encryption_key(request.user.username)
#
#     # Proceed to respond with the username and encryption key
#     return Response({
#       "username": request.user.username,
#     })
#
#   def post(self, request, *args, **kwargs):
#     # Ensure the encryption key is generated before handling the POST request
#     if not self.encryption_key:
#       self.generate_encryption_key(request.user.username)
#
#     # Log the received data and encryption key
#     print(request.data)
#     print(f"Authenticated username: {request.user.username}")
#     print(f"Encryption Key: {self.encryption_key}")
#
#     # Processing the posted message
#     message = request.data.get('message')
#     print(message)  # Logging the received message
#     chat_response = 'Hello'  # A simple fixed response for demonstration
#     return Response({"chat_response": chat_response})

# class ChatPdfView(generics.RetrieveAPIView):
#   permission_classes = [permissions.IsAuthenticated]
#   # Initialization right here to ensure it's easily visible for demonstration
#
#   def get(self, request, *args, **kwargs):
#
#     # Proceed to respond with the username and encryption key
#     return Response({
#       "username": request.user.username,
#     })
#
#   def post(self, request, *args, **kwargs):
#     try :
#       persist_dir = "./vector_databases/" + request.user.username + "_chroma_db"
#
#       vectordb = Chroma(persist_directory=persist_dir, embedding_function=embed_model)
#       # print(vectordb)
#       print(vectordb._collection.count())
#     except :
#       print('No database found for ',request.user.username)
#
#     print(request.data)
#     print(f"Authenticated username: {request.user.username}")
#
#     # Processing the posted message
#     message = request.data.get('message')
#     print(message)  # Logging the received message
#     chat_response = 'Hello'  # A simple fixed response for demonstration
#     return Response({"chat_response": chat_response})

from langchain_community.llms import HuggingFaceEndpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
)

from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, RetrievalQA, ConversationalRetrievalChain

# [INST] <> for other model

template = """
[INST] <>
Act as a legal expert. Use the following information to answer the question at the end.
<>

{context}

{question} [/INST]
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

class ChatPdfView(generics.RetrieveAPIView):
  permission_classes = [permissions.IsAuthenticated]
  # Initialization right here to ensure it's easily visible for demonstration

  def get(self, request, *args, **kwargs):

    # Proceed to respond with the username and encryption key
    return Response({
      "username": request.user.username,
    })

  def post(self, request, *args, **kwargs):
    try :
      persist_dir = "./vector_databases/" + request.user.username + "_chroma_db"

      vectordb = Chroma(persist_directory=persist_dir, embedding_function=embed_model)
      # print(vectordb)
      print(vectordb._collection.count())
    except :
      print('No database found for ',request.user.username)

    qa_chain = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=vectordb.as_retriever(search_kwargs={"k": 1}),
      return_source_documents=True,
      # chain_type_kwargs={"prompt": prompt},
    )

    # query = "What is this document about?"

    query = request.data.get('message')
    result_ = qa_chain(
      query
    )
    result = result_["result"].strip()
    print('result          :',result)
    print(result_.keys())
    source_doc = result_["source_documents"]
    print('source_doc              :',source_doc)
    print(request.data)
    print(f"Authenticated username: {request.user.username}")

    # Processing the posted message
    # message = request.data.get('message')
    print(query)  # Logging the received message
    # chat_response = 'Hello'  # A simple fixed response for demonstration
    chat_response = result  # A simple fixed response for demonstration
    return Response({"chat_response": chat_response})
