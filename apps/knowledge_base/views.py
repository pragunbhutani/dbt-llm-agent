from django.shortcuts import render
from rest_framework import viewsets

# Update imports
from .models import Model
from .serializers import ModelSerializer

# Create your views here.


class ModelViewSet(viewsets.ModelViewSet):
    """API endpoint that allows Models to be viewed or edited."""

    queryset = Model.objects.all().order_by("-created_at")
    serializer_class = ModelSerializer
    # permission_classes = [permissions.IsAuthenticated] # Example
