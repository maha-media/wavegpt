from django.urls import path
from . import views

urlpatterns = [
    path('stream/', views.stream),
    path('transactions/', views.transactions),
    path('regime/', views.regime),
    path('portfolio/', views.portfolio),
    path('rebalances/', views.rebalances),
    path('chat/', views.chat),
]
