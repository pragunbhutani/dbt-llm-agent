from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import QuestionViewSet, AskQuestionView

router = DefaultRouter()
router.register(r"questions", QuestionViewSet, basename="question")

urlpatterns = [
    path("", include(router.urls)),
    path("ask/", AskQuestionView.as_view(), name="ask-question"),
]
