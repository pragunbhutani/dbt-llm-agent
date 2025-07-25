from django.shortcuts import render
from rest_framework import generics
from .models import WaitlistEntry
from .serializers import WaitlistEntrySerializer

# Create your views here.


class WaitlistEntryCreateView(generics.CreateAPIView):
    queryset = WaitlistEntry.objects.all()
    serializer_class = WaitlistEntrySerializer
