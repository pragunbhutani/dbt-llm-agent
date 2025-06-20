from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import QuestionViewSet, ConversationViewSet, AskQuestionView

router = DefaultRouter(trailing_slash=True)
router.register(r"questions", QuestionViewSet, basename="question")
router.register(r"conversations", ConversationViewSet, basename="conversation")

urlpatterns = [
    path("", include(router.urls)),
    path("ask/", AskQuestionView.as_view(), name="ask-question"),
]
