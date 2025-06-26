from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DbtProjectViewSet
from .dashboard_api import DashboardStatsAPIView

router = DefaultRouter(trailing_slash=True)
router.register(r"projects", DbtProjectViewSet, basename="dbt-project")

app_name = "data_sources"

urlpatterns = [
    path("", include(router.urls)),
    path("dashboard-stats/", DashboardStatsAPIView.as_view(), name="dashboard_stats"),
]
