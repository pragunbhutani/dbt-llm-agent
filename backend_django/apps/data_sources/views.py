from django.shortcuts import render
from rest_framework import viewsets
from .models import DbtProject
from .serializers import DbtProjectSerializer
from rest_framework.permissions import IsAuthenticated

# Create your views here.


class DbtProjectViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows dbt projects to be viewed or edited.
    """

    queryset = DbtProject.objects.all()
    serializer_class = DbtProjectSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """
        This view should return a list of all the dbt projects
        for the currently authenticated user's organisation.
        """
        user = self.request.user
        return DbtProject.objects.for_organisation(user.organisation)

    def perform_create(self, serializer):
        """
        Automatically associate the dbt project with the user's organisation.
        """
        serializer.save(organisation=self.request.user.organisation)
