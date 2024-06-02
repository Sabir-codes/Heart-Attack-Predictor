from django.contrib import admin
from django.urls import path, include
from predictor import views

urlpatterns = [
    path('', views.index, name='predict')

]
