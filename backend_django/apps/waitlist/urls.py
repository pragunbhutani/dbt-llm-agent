from django.urls import path
from .views import WaitlistEntryCreateView

urlpatterns = [
    path("", WaitlistEntryCreateView.as_view(), name="waitlist-create"),
]
