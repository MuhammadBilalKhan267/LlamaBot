from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("base/", views.base, name="base"),
    path("chat/<int:id>", views.chat, name="chat"),
    path("add_message/", views.add_message, name="add_message"),
    path("add_file/", views.add_file, name="add_file"),
    path("add_chat/", views.add_chat, name="add_chat"),
    path("delete/", views.delete_chat, name="delete_chat"),
]