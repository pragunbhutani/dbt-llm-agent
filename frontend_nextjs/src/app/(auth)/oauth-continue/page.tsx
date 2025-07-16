"use client";

import { useEffect, useState, Suspense } from "react";
import { useSession } from "next-auth/react";
import { useSearchParams, useRouter } from "next/navigation";

function OAuthContinueContent() {
  const { data: session, status } = useSession();
  const searchParams = useSearchParams();
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    const continueOAuthFlow = async () => {
      if (status === "loading") return;

      const next = searchParams.get("next");

      if (!next) {
        setError("Missing OAuth continuation URL");
        return;
      }

      if (!session?.accessToken) {
        // User is not authenticated, redirect to sign in with the original next URL
        router.push(`/signin?next=${encodeURIComponent(next)}`);
        return;
      }

      // User is authenticated, continue the OAuth flow
      setIsProcessing(true);

      try {
        // Parse the auth_request_id from the next URL
        const nextUrl = new URL(next);
        const authRequestId = nextUrl.searchParams.get("auth_request_id");

        if (!authRequestId) {
          throw new Error("Missing auth_request_id in OAuth continuation URL");
        }

        // Call the MCP callback endpoint with the user's token
        const callbackUrl = new URL(next);
        callbackUrl.searchParams.set("user_token", session.accessToken);

        // Redirect to the callback URL
        window.location.href = callbackUrl.toString();
      } catch (err) {
        console.error("OAuth continuation error:", err);
        setError(
          err instanceof Error ? err.message : "OAuth continuation failed"
        );
        setIsProcessing(false);
      }
    };

    continueOAuthFlow();
  }, [session, status, searchParams, router]);

  if (status === "loading" || isProcessing) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="max-w-md w-full space-y-8 p-8">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <h2 className="mt-6 text-3xl font-extrabold text-gray-900">
              Completing OAuth Authorization
            </h2>
            <p className="mt-2 text-sm text-gray-600">
              Please wait while we complete your authorization...
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="max-w-md w-full space-y-8 p-8">
          <div className="text-center">
            <div className="mx-auto h-12 w-12 text-red-600">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"
                />
              </svg>
            </div>
            <h2 className="mt-6 text-3xl font-extrabold text-gray-900">
              OAuth Authorization Error
            </h2>
            <p className="mt-2 text-sm text-gray-600">{error}</p>
            <button
              onClick={() => router.push("/dashboard")}
              className="mt-4 w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              Return to Dashboard
            </button>
          </div>
        </div>
      </div>
    );
  }

  return null;
}

export default function OAuthContinuePage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
          <div className="max-w-md w-full space-y-8 p-8">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
              <h2 className="mt-6 text-3xl font-extrabold text-gray-900">
                Loading...
              </h2>
              <p className="mt-2 text-sm text-gray-600">
                Please wait while we prepare your OAuth authorization...
              </p>
            </div>
          </div>
        </div>
      }
    >
      <OAuthContinueContent />
    </Suspense>
  );
}
