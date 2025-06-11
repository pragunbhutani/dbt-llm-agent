from django.shortcuts import render
import logging
from rest_framework import status, views
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from .serializers import UserRegistrationSerializer, UserSerializer
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework import generics
from .models import OrganisationSettings
from .serializers import OrganisationSettingsSerializer

logger = logging.getLogger(__name__)

# Create your core views here, if needed later.

# Views previously in this file moved to:
# - ModelViewSet -> apps.knowledge_base.views
# - QuestionViewSet, AskQuestionView -> apps.workflows.views
# - ModelEmbeddingViewSet -> apps.embeddings.views

# --- End of file --- Ensure everything below this line is removed ---


class UserRegistrationView(views.APIView):
    permission_classes = [AllowAny]
    # serializer_class = UserRegistrationSerializer # Not needed here as we instantiate manually

    def post(self, request, *args, **kwargs):
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()

            # Generate JWT tokens for the new user
            refresh = RefreshToken.for_user(user)

            # Get user data for the response using UserSerializer
            user_data = UserSerializer(user, context={"request": request}).data

            return Response(
                {
                    "refresh": str(refresh),
                    "access": str(refresh.access_token),
                    "user": user_data,
                },
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserView(views.APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data)


class OrganisationSettingsView(generics.RetrieveUpdateAPIView):
    """
    API endpoint for viewing and editing the organisation's settings.
    """

    serializer_class = OrganisationSettingsSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        """
        Retrieve or create the settings object for the user's organisation.
        """
        organisation = self.request.user.organisation
        # get_or_create returns a tuple (object, created_boolean)
        settings, created = OrganisationSettings.objects.get_or_create(
            organisation=organisation
        )
        return settings
