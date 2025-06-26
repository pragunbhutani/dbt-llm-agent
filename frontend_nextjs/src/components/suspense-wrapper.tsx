import { Suspense } from "react";
import { Loader2 } from "lucide-react";

interface SuspenseWrapperProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  loadingText?: string;
}

export function SuspenseWrapper({
  children,
  fallback,
  loadingText = "Loading...",
}: SuspenseWrapperProps) {
  const defaultFallback = (
    <div className="flex flex-col items-center justify-center min-h-32 space-y-4">
      <Loader2 className="h-6 w-6 animate-spin text-indigo-600" />
      <p className="text-sm text-gray-600">{loadingText}</p>
    </div>
  );

  return <Suspense fallback={fallback || defaultFallback}>{children}</Suspense>;
}
