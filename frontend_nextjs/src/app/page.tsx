"use client";

import { useAuth } from "@/lib/useAuth";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  ArrowRight,
  Database,
  MessageSquare,
  GitBranch,
  BarChart3,
  Slack,
  Zap,
  Clock,
  Users,
  CheckCircle,
  CheckIcon,
  Settings,
  Brain,
} from "lucide-react";

export default function Home() {
  const { isAuthenticated, isLoading } = useAuth();

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
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
              <Database className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-gray-900">Ragstar</span>
          </div>
          <nav className="hidden md:flex items-center space-x-8">
            <a
              href="#features"
              className="text-gray-600 hover:text-gray-900 transition-colors"
            >
              Features
            </a>
            <a
              href="#how-it-works"
              className="text-gray-600 hover:text-gray-900 transition-colors"
            >
              How it Works
            </a>
            <a
              href="#benefits"
              className="text-gray-600 hover:text-gray-900 transition-colors"
            >
              Benefits
            </a>
            <a
              href="#pricing"
              className="text-gray-600 hover:text-gray-900 transition-colors"
            >
              Pricing
            </a>
            {isAuthenticated ? (
              <Link href="/dashboard">
                <Button className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white">
                  Go to Dashboard
                </Button>
              </Link>
            ) : (
              <div className="flex items-center space-x-4">
                <Link href="/signin">
                  <Button
                    variant="ghost"
                    className="text-gray-600 hover:text-gray-900"
                  >
                    Sign In
                  </Button>
                </Link>
                <Link href="/signup">
                  <Button className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white">
                    Get Started
                  </Button>
                </Link>
              </div>
            )}
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20 px-4">
        <div className="container mx-auto text-center max-w-4xl">
          <Badge
            variant="secondary"
            className="mb-4 bg-blue-100 text-blue-800 border-blue-200"
          >
            AI-Powered Data Analytics
          </Badge>
          <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6 leading-tight">
            Stop Spending 50% of Your Time on{" "}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">
              Data Requests
            </span>
          </h1>
          <p className="text-xl text-gray-600 mb-8 leading-relaxed">
            Ragstar connects to your dbt projects and automatically answers
            operational data questions, freeing your engineering team to focus
            on what matters most.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            {isAuthenticated ? (
              <Link href="/dashboard">
                <Button
                  size="lg"
                  className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-8 py-3"
                >
                  Go to Dashboard
                  <ArrowRight className="ml-2 w-5 h-5" />
                </Button>
              </Link>
            ) : (
              <>
                <Link href="/signup">
                  <Button
                    size="lg"
                    className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-8 py-3"
                  >
                    Get Early Access
                    <ArrowRight className="ml-2 w-5 h-5" />
                  </Button>
                </Link>
                <Button
                  size="lg"
                  variant="outline"
                  className="bg-white text-gray-900 border-gray-300 px-8 py-3"
                >
                  <a href="#how-it-works">Watch Demo</a>
                </Button>
              </>
            )}
          </div>
        </div>
      </section>

      {/* Problem Section */}
      <section className="py-16 px-4 bg-gray-50">
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
      <section id="how-it-works" className="py-16 px-4">
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

      {/* Features Section */}
      <section id="features" className="py-16 px-4 bg-gray-50">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Flexible & Powerful Integration
            </h2>
            <p className="text-lg text-gray-600">
              Access your data insights through your preferred channels and
              tools
            </p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="bg-white border-gray-200 text-center">
              <CardHeader>
                <Slack className="w-8 h-8 text-purple-600 mx-auto mb-2" />
                <CardTitle className="text-lg">Slack Integration</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600 text-sm">
                  Ask questions directly in your team channels
                </p>
              </CardContent>
            </Card>
            <Card className="bg-white border-gray-200 text-center">
              <CardHeader>
                <Brain className="w-8 h-8 text-blue-600 mx-auto mb-2" />
                <CardTitle className="text-lg">Any LLM Provider</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600 text-sm">
                  Use OpenAI, Anthropic, or any LLM provider of your choice
                </p>
              </CardContent>
            </Card>
            <Card className="bg-white border-gray-200 text-center">
              <CardHeader>
                <BarChart3 className="w-8 h-8 text-green-600 mx-auto mb-2" />
                <CardTitle className="text-lg">Metabase Integration</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600 text-sm">
                  Connect with Metabase for enhanced visualization
                </p>
              </CardContent>
            </Card>
            <Card className="bg-white border-gray-200 text-center">
              <CardHeader>
                <Database className="w-8 h-8 text-orange-600 mx-auto mb-2" />
                <CardTitle className="text-lg">Snowflake & More</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600 text-sm">
                  Connect directly to Snowflake and other data warehouses
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section id="benefits" className="py-16 px-4">
        <div className="container mx-auto max-w-4xl">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Reclaim Your Team&apos;s Time
            </h2>
            <p className="text-lg text-gray-600">
              See the immediate impact on your data engineering productivity
            </p>
          </div>
          <div className="grid md:grid-cols-2 gap-8">
            <div className="space-y-6">
              <div className="flex items-start space-x-3">
                <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0 mt-1" />
                <div>
                  <h3 className="font-semibold text-gray-900 mb-1">
                    50% Time Savings
                  </h3>
                  <p className="text-gray-600">
                    Eliminate repetitive data request handling
                  </p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0 mt-1" />
                <div>
                  <h3 className="font-semibold text-gray-900 mb-1">
                    Instant Responses
                  </h3>
                  <p className="text-gray-600">
                    No more waiting days for simple data questions
                  </p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0 mt-1" />
                <div>
                  <h3 className="font-semibold text-gray-900 mb-1">
                    Better Onboarding
                  </h3>
                  <p className="text-gray-600">
                    New engineers learn your data models faster
                  </p>
                </div>
              </div>
            </div>
            <div className="space-y-6">
              <div className="flex items-start space-x-3">
                <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0 mt-1" />
                <div>
                  <h3 className="font-semibold text-gray-900 mb-1">
                    Consistent Accuracy
                  </h3>
                  <p className="text-gray-600">
                    AI ensures consistent, reliable data analysis
                  </p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0 mt-1" />
                <div>
                  <h3 className="font-semibold text-gray-900 mb-1">
                    Self-Service Analytics
                  </h3>
                  <p className="text-gray-600">
                    Stakeholders get answers without bothering your team
                  </p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0 mt-1" />
                <div>
                  <h3 className="font-semibold text-gray-900 mb-1">
                    Focus on Innovation
                  </h3>
                  <p className="text-gray-600">
                    Spend time building, not answering questions
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <div id="pricing" className="bg-gray-50 py-24 sm:py-32">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl sm:text-center">
            <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
              Simple, transparent pricing
            </h2>
            <p className="mt-6 text-lg leading-8 text-gray-600">
              Choose the plan that works best for your team
            </p>
          </div>
          <div className="mx-auto mt-16 grid max-w-lg grid-cols-1 gap-y-6 sm:mt-20 sm:gap-y-0 lg:max-w-4xl lg:grid-cols-2">
            {/* Free Plan */}
            <div className="rounded-3xl bg-white p-8 ring-1 ring-gray-200 sm:p-10 lg:mx-8 lg:rounded-r-none">
              <h3 className="text-base font-semibold leading-7 text-indigo-600">
                Open Source
              </h3>
              <p className="mt-4 flex items-baseline gap-x-2">
                <span className="text-5xl font-bold tracking-tight text-gray-900">
                  Free
                </span>
              </p>
              <p className="mt-6 text-base leading-7 text-gray-600">
                Perfect for individual developers and small teams getting
                started.
              </p>
              <ul className="mt-8 space-y-3 text-sm leading-6 text-gray-600">
                <li className="flex gap-x-3">
                  <CheckIcon className="h-6 w-5 flex-none text-indigo-600" />
                  Self-hosted deployment
                </li>
                <li className="flex gap-x-3">
                  <CheckIcon className="h-6 w-5 flex-none text-indigo-600" />
                  Connect unlimited dbt projects
                </li>
                <li className="flex gap-x-3">
                  <CheckIcon className="h-6 w-5 flex-none text-indigo-600" />
                  Basic Slack integration
                </li>
                <li className="flex gap-x-3">
                  <CheckIcon className="h-6 w-5 flex-none text-indigo-600" />
                  Community support
                </li>
              </ul>
              <a
                href="https://github.com/pragunbhutani/ragstar"
                target="_blank"
                rel="noopener noreferrer"
                className="mt-8 block rounded-md px-3.5 py-2.5 text-center text-sm font-semibold text-indigo-600 ring-1 ring-inset ring-indigo-200 hover:ring-indigo-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
              >
                View on GitHub
              </a>
            </div>

            {/* Paid Plan */}
            <div className="rounded-3xl bg-indigo-600 p-8 ring-1 ring-indigo-600 sm:p-10 lg:mx-8 lg:rounded-l-none">
              <h3 className="text-base font-semibold leading-7 text-white">
                Ragstar Cloud
              </h3>
              <p className="mt-4 flex items-baseline gap-x-2">
                <span className="text-5xl font-bold tracking-tight text-white">
                  $99
                </span>
                <span className="text-base text-indigo-200">/month</span>
              </p>
              <p className="mt-6 text-base leading-7 text-indigo-200">
                Fully managed cloud service with premium features and support.
              </p>
              <ul className="mt-8 space-y-3 text-sm leading-6 text-indigo-200">
                <li className="flex gap-x-3">
                  <CheckIcon className="h-6 w-5 flex-none text-white" />
                  Fully managed hosting
                </li>
                <li className="flex gap-x-3">
                  <CheckIcon className="h-6 w-5 flex-none text-white" />
                  Advanced analytics & insights
                </li>
                <li className="flex gap-x-3">
                  <CheckIcon className="h-6 w-5 flex-none text-white" />
                  Premium integrations
                </li>
                <li className="flex gap-x-3">
                  <CheckIcon className="h-6 w-5 flex-none text-white" />
                  Priority support
                </li>
                <li className="flex gap-x-3">
                  <CheckIcon className="h-6 w-5 flex-none text-white" />
                  Team collaboration features
                </li>
              </ul>
              {isAuthenticated ? (
                <Link
                  href="/dashboard"
                  className="mt-8 block rounded-md bg-white px-3.5 py-2.5 text-center text-sm font-semibold text-indigo-600 shadow-sm hover:bg-indigo-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
                >
                  Go to Dashboard
                </Link>
              ) : (
                <Link
                  href="/signup"
                  className="mt-8 block rounded-md bg-white px-3.5 py-2.5 text-center text-sm font-semibold text-indigo-600 shadow-sm hover:bg-indigo-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
                >
                  Start free trial
                </Link>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <section className="py-16 px-4 bg-gradient-to-r from-blue-600 to-purple-600">
        <div className="container mx-auto max-w-4xl text-center">
          <h2 className="text-3xl font-bold text-white mb-4">
            Ready to Transform Your Data Team?
          </h2>
          <p className="text-xl text-blue-100 mb-8">
            {isAuthenticated
              ? "Continue building your data analytics workflow in your dashboard."
              : "Join the teams already using Ragstar to make data-driven decisions faster than ever."}
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            {isAuthenticated ? (
              <Link href="/dashboard">
                <Button
                  size="lg"
                  className="bg-white text-blue-600 hover:bg-gray-100 px-8 py-3"
                >
                  Go to Dashboard
                  <ArrowRight className="ml-2 w-5 h-5" />
                </Button>
              </Link>
            ) : (
              <>
                <Link href="/signup">
                  <Button
                    size="lg"
                    className="bg-white text-blue-600 hover:bg-gray-100 px-8 py-3"
                  >
                    Get Early Access
                    <ArrowRight className="ml-2 w-5 h-5" />
                  </Button>
                </Link>
                <Button
                  size="lg"
                  variant="outline"
                  className="border-white text-white hover:bg-white hover:text-blue-600 px-8 py-3"
                >
                  <a href="#pricing">View Pricing</a>
                </Button>
              </>
            )}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 bg-gray-900 text-white">
        <div className="container mx-auto max-w-6xl">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                  <Database className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-bold">Ragstar</span>
              </div>
              <p className="text-gray-400">
                AI-powered data analytics for modern data teams using dbt.
              </p>
            </div>
            <div>
              <h3 className="font-semibold mb-4">Product</h3>
              <ul className="space-y-2 text-gray-400">
                <li>
                  <a
                    href="#features"
                    className="hover:text-white transition-colors"
                  >
                    Features
                  </a>
                </li>
                <li>
                  <a
                    href="#how-it-works"
                    className="hover:text-white transition-colors"
                  >
                    How it Works
                  </a>
                </li>
                <li>
                  <a
                    href="#pricing"
                    className="hover:text-white transition-colors"
                  >
                    Pricing
                  </a>
                </li>
                <li>
                  <a
                    href="https://github.com/pragunbhutani/ragstar"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-white transition-colors"
                  >
                    Documentation
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-4">Company</h3>
              <ul className="space-y-2 text-gray-400">
                <li>
                  <a
                    href="https://github.com/pragunbhutani/ragstar"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-white transition-colors"
                  >
                    About
                  </a>
                </li>
                <li>
                  <a
                    href="https://github.com/pragunbhutani/ragstar"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-white transition-colors"
                  >
                    GitHub
                  </a>
                </li>
                <li>
                  <a
                    href="https://github.com/pragunbhutani/ragstar/issues"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-white transition-colors"
                  >
                    Issues
                  </a>
                </li>
                <li>
                  <a
                    href="https://github.com/pragunbhutani/ragstar/discussions"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-white transition-colors"
                  >
                    Discussions
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-4">Support</h3>
              <ul className="space-y-2 text-gray-400">
                <li>
                  <a
                    href="https://github.com/pragunbhutani/ragstar"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-white transition-colors"
                  >
                    Help Center
                  </a>
                </li>
                <li>
                  <a
                    href="https://github.com/pragunbhutani/ragstar/discussions"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-white transition-colors"
                  >
                    Community
                  </a>
                </li>
                <li>
                  <a
                    href="https://github.com/pragunbhutani/ragstar/issues"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-white transition-colors"
                  >
                    Report Issues
                  </a>
                </li>
                <li>
                  <a
                    href="https://github.com/pragunbhutani/ragstar"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-white transition-colors"
                  >
                    Privacy
                  </a>
                </li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2024 Ragstar. Open source data analytics platform.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
