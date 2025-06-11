from django.shortcuts import render
from rest_framework import viewsets, permissions
from rest_framework.exceptions import PermissionDenied

# Update imports
from .models import Model
from .serializers import ModelSerializer

# Create your views here.


class ModelViewSet(viewsets.ModelViewSet):
    """API endpoint that allows Models to be viewed or edited, scoped by organisation."""

    serializer_class = ModelSerializer
    permission_classes = [permissions.IsAuthenticated]  # Ensure user is authenticated

    def get_queryset(self):
        user = self.request.user
        if hasattr(user, "organisation") and user.organisation:
            # Use the custom manager method if available and preferred,
            # otherwise filter directly.
            # Assumes the default manager on Model is OrganisationScopedManager or similar
            return Model.objects.for_organisation(user.organisation).order_by(
                "-created_at"
            )
        # If user has no organisation, or for other reasons, return no objects
        return Model.objects.none()

    def perform_create(self, serializer):
        user = self.request.user
        if hasattr(user, "organisation") and user.organisation:
            serializer.save(organisation=user.organisation)
        else:
            # This case should ideally not be reached if permissions are set correctly
            # and all authenticated users have an organisation.
            raise PermissionDenied("User is not associated with an organisation.")
