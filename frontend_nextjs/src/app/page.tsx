"use client";

import { useAuth } from "@/lib/useAuth";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  MessageSquare,
  GitBranch,
  BarChart3,
  Zap,
  Clock,
  Users,
} from "lucide-react";
import PricingSection from "@/components/landing/pricing-section";
import LandingHero from "@/components/landing/hero";
import FeatureList from "@/components/landing/feature-list";
import CtaSection from "@/components/landing/cta-section";
import Testimonials from "@/components/landing/testimonials";
import Footer from "@/components/landing/footer";
import BentoFeatures from "@/components/landing/bento-features";

export default function Home() {
  const { isLoading } = useAuth();

  if (isLoading) {
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
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-white">
      {/* Hero Section (new) */}
      <LandingHero />

      {/* Problem Section */}
      <section className="py-12 px-4 bg-gray-50">
        <div className="container mx-auto max-w-4xl">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              The Data Request Bottleneck
            </h2>
            <p className="text-lg text-gray-600">
              Data teams are drowning in operational requests that could be
              automated
            </p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            <Card className="bg-white border-gray-200">
              <CardHeader>
                <Clock className="w-8 h-8 text-red-500 mb-2" />
                <CardTitle className="text-lg">50% Time Drain</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600">
                  Half your engineer&apos;s bandwidth goes to answering
                  repetitive data questions instead of building.
                </p>
              </CardContent>
            </Card>
            <Card className="bg-white border-gray-200">
              <CardHeader>
                <MessageSquare className="w-8 h-8 text-orange-500 mb-2" />
                <CardTitle className="text-lg">
                  Constant Interruptions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600">
                  Product, marketing, and executives need reports, lookalike
                  audiences, and one-time analyses.
                </p>
              </CardContent>
            </Card>
            <Card className="bg-white border-gray-200">
              <CardHeader>
                <Users className="w-8 h-8 text-blue-500 mb-2" />
                <CardTitle className="text-lg">Slow Onboarding</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600">
                  New engineers struggle to understand complex dbt models and
                  data relationships.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Solution Section */}
      <section id="how-it-works" className="py-12 px-4">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Your AI Data Analyst
            </h2>
            <p className="text-lg text-gray-600">
              Connect your dbt projects and let AI handle the questions
              automatically
            </p>
          </div>
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <div className="space-y-6">
                <div className="flex items-start space-x-4">
                  <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                    <GitBranch className="w-5 h-5 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      Connect Your dbt Projects
                    </h3>
                    <p className="text-gray-600">
                      Seamlessly integrate with dbt Cloud or dbt Core GitHub
                      repositories to build your knowledge base.
                    </p>
                  </div>
                </div>
                <div className="flex items-start space-x-4">
                  <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center flex-shrink-0">
                    <Zap className="w-5 h-5 text-purple-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      AI-Powered Analysis
                    </h3>
                    <p className="text-gray-600">
                      Advanced AI workflows understand your data models and
                      generate accurate queries and insights.
                    </p>
                  </div>
                </div>
                <div className="flex items-start space-x-4">
                  <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center flex-shrink-0">
                    <BarChart3 className="w-5 h-5 text-green-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      Instant Visualizations
                    </h3>
                    <p className="text-gray-600">
                      Get charts, queries, and insights delivered instantly, or
                      receive the SQL to run yourself.
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl p-8 text-white">
              <div className="space-y-4">
                <div className="flex items-center space-x-2 text-green-400">
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  <span className="text-sm">Connected to dbt Cloud</span>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-sm text-gray-300 mb-2">
                    💬 Product Manager asks:
                  </p>
                  <p className="text-white">
                    &quot;What&apos;s our user retention rate for the last
                    quarter?&quot;
                  </p>
                </div>
                <div className="bg-blue-600 rounded-lg p-4">
                  <p className="text-sm text-blue-200 mb-2">
                    🤖 Ragstar AI responds:
                  </p>
                  <p className="text-white text-sm">
                    &quot;Q4 user retention is 73.2%. Here&apos;s the breakdown
                    by cohort and the SQL query used...&quot;
                  </p>
                  <div className="mt-3 bg-blue-700 rounded p-2 text-xs font-mono">
                    SELECT cohort_month, retention_rate FROM
                    user_retention_analysis...
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Feature Highlights */}
      <div className="bg-gray-50">
        <BentoFeatures />
      </div>
      <FeatureList />

      {/* Pricing Section */}
      <div className="bg-gray-50">
        <PricingSection />
      </div>

      {/* Testimonials & CTA */}
      <Testimonials />
      <CtaSection />

      {/* Footer */}
      <Footer />
    </div>
  );
}
