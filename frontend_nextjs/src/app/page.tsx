"use client";

import { useSession } from "next-auth/react";
import Navbar from "@/components/landing/navbar";
import LandingHero from "@/components/landing/hero";
import PainPoints from "@/components/landing/pain-points";
import Solution from "@/components/landing/solution";
import BentoFeatures from "@/components/landing/bento-features";
import PricingSection from "@/components/landing/pricing-section";
import Installation from "@/components/landing/installation";
import CtaSection from "@/components/landing/cta-section";
import Testimonials from "@/components/landing/testimonials";
import Footer from "@/components/landing/footer";
import Stats from "@/components/landing/stats";

export default function Home() {
  const { status } = useSession();

  // Only show loading if we're actively loading a session
  // Don't show loading for unauthenticated users
  if (status === "loading") {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-center">
          <div className="h-32 w-32 animate-spin rounded-full border-b-2 border-indigo-600 mx-auto"></div>
          <p className="mt-4 text-lg text-gray-600">Loading Ragstar...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <Navbar />

      {/* Hero Section */}
      <LandingHero />

      {/* Pain Points Section */}
      <PainPoints />

      {/* Solution Section */}
      <Solution />

      {/* Stats Section */}
      <Stats />

      {/* Features Section */}
      <BentoFeatures />

      {/* Pricing Section */}
      <PricingSection />

      {/* Installation Instructions */}
      <Installation />

      {/* Testimonials */}
      {/* <Testimonials /> */}

      {/* Final CTA */}
      <CtaSection />

      {/* Footer */}
      <Footer />
    </div>
  );
}
