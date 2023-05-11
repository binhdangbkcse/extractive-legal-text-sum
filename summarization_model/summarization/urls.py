from django.urls import path
from .views import *
from summarization import views

urlpatterns = [
    path('get-sum', views.summarization, name='get-sum')
]