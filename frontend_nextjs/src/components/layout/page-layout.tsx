import React from "react";
import Heading from "@/components/heading";
import { cn } from "@/lib/utils";

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  /**
   * Optional right-hand side content (e.g., buttons, status badges).
   */
  actions?: React.ReactNode;
  /**
   * Additional class names for the header wrapper.
   */
  className?: string;
}

export function PageHeader({
  title,
  subtitle,
  actions,
  className,
}: PageHeaderProps) {
  return (
    <header
      className={cn(
        "sticky top-0 z-10 flex items-center justify-between border-b border-gray-200 bg-white px-2 py-2 lg:px-4",
        className
      )}
    >
      <Heading title={title} subtitle={subtitle} />
      {actions && <div className="ml-4 flex-shrink-0">{actions}</div>}
    </header>
  );
}

interface PageBodyProps {
  children: React.ReactNode;
  className?: string;
}

export function PageBody({ children, className }: PageBodyProps) {
  return (
    <section
      className={cn(
        "flex-1 overflow-auto bg-gray-50 px-2 py-4 lg:px-4",
        className
      )}
    >
      {children}
    </section>
  );
}

interface PageLayoutProps extends PageHeaderProps {
  children: React.ReactNode;
}

/**
 * Convenience wrapper that composes PageHeader + PageBody for the common
 * dashboard page layout.
 */
export default function PageLayout({
  title,
  subtitle,
  actions,
  children,
}: PageLayoutProps) {
  return (
    <div className="flex h-full flex-col min-h-screen">
      <PageHeader title={title} subtitle={subtitle} actions={actions} />
      <PageBody>{children}</PageBody>
    </div>
  );
}
