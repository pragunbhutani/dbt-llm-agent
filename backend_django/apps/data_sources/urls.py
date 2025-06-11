from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DbtProjectViewSet

router = DefaultRouter()
router.register(r"dbt-projects", DbtProjectViewSet, basename="dbtproject")

urlpatterns = [
    path("", include(router.urls)),
]
