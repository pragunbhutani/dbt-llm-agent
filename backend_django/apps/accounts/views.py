import logging

from django.contrib.auth import get_user_model
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework import generics, permissions, status, views
from rest_framework.exceptions import NotFound
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView

from .models import Organisation, OrganisationSettings
from .serializers import (
    OrganisationSettingsSerializer,
    UserRegistrationSerializer,
    UserSerializer,
)

logger = logging.getLogger(__name__)
User = get_user_model()


@method_decorator(csrf_exempt, name="dispatch")
class UserRegistrationView(generics.CreateAPIView):
    permission_classes = [AllowAny]
    serializer_class = UserRegistrationSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        refresh = RefreshToken.for_user(user)
        user_data = UserSerializer(user, context={"request": request}).data

        return Response(
            {
                "refresh": str(refresh),
                "access": str(refresh.access_token),
                "user": user_data,
            },
            status=status.HTTP_201_CREATED,
        )


@method_decorator(csrf_exempt, name="dispatch")
class CustomTokenObtainPairView(TokenObtainPairView):
    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)
        if response.status_code == 200:
            user = User.objects.get(email=request.data["email"])
            user_data = UserSerializer(user, context={"request": request}).data
            response.data["user"] = user_data
        return response


class UserView(views.APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data)


class OrganisationSettingsView(generics.RetrieveUpdateAPIView):
    """
    Retrieve or update the settings for the user's organisation.
    """

    serializer_class = OrganisationSettingsSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        user = self.request.user
        if not hasattr(user, "organisation") or not user.organisation:
            raise NotFound(detail="User is not associated with an organisation.")

        settings, _ = OrganisationSettings.objects.get_or_create(
            organisation=user.organisation
        )
        return settings

    def update(self, request, *args, **kwargs):
        """Enhanced update method to provide feedback about Slack team detection."""
        partial = kwargs.pop("partial", False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)

        self.perform_update(serializer)

        # Return standard response
        response_data = serializer.data

        return Response(response_data)
