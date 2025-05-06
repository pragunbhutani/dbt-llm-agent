from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ModelEmbeddingViewSet

router = DefaultRouter()
router.register(r"model-embeddings", ModelEmbeddingViewSet, basename="modelembedding")

urlpatterns = [
    path("", include(router.urls)),
]
