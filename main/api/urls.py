from django.urls import path
from .views  import detectionAPI

urlpatterns = [path('api', detectionAPI.as_view())]