
from django.contrib import admin
from django.urls import path
from . import views

from . import views
urlpatterns = [
    path("", views.home, name="home"),
    path("predict/", views.predict, name="predict"),
]