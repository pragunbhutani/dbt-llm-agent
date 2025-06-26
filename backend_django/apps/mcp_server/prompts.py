"""MCP prompts for guiding LLM clients on how to use Ragstar effectively."""

from typing import Any, Dict, List, Optional


def get_prompts() -> List[Dict[str, Any]]:
    """Return list of available MCP prompts."""
    return [
        {
            "name": "ragstar_introduction",
            "description": "Learn about Ragstar and its capabilities for dbt analytics",
            "arguments": [],
        },
        {
            "name": "data_analysis_guidance",
            "description": "Get guidance on how to analyze data and ask effective questions using Ragstar",
            "arguments": [],
        },
        {
            "name": "model_exploration_guide",
            "description": "Learn how to explore and understand dbt models in your knowledge base",
            "arguments": [
                {
                    "name": "focus_area",
                    "description": "Specific area to focus on (e.g., 'searching', 'lineage', 'documentation')",
                    "required": False,
                }
            ],
        },
        {
            "name": "best_practices",
            "description": "Get best practices for working with dbt analytics and Ragstar",
            "arguments": [
                {
                    "name": "topic",
                    "description": "Specific topic (e.g., 'querying', 'model_naming', 'documentation')",
                    "required": False,
                }
            ],
        },
    ]


async def handle_prompt_request(
    name: str, arguments: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Handle MCP prompt requests."""
    arguments = arguments or {}

    if name == "ragstar_introduction":
        return _get_ragstar_introduction()
    elif name == "data_analysis_guidance":
        return _get_data_analysis_guidance()
    elif name == "model_exploration_guide":
        return _get_model_exploration_guide(arguments.get("focus_area"))
    elif name == "best_practices":
        return _get_best_practices(arguments.get("topic"))
    else:
        return {
            "error": f"Unknown prompt: {name}",
            "content": [{"type": "text", "text": f"Prompt '{name}' is not available."}],
        }


def _get_ragstar_introduction() -> Dict[str, Any]:
    """Introduction to Ragstar and its capabilities."""
    content = """# Welcome to Ragstar! 🌟

Ragstar is an AI-powered data analyst designed specifically for teams working with **dbt** (data build tool). I help you understand, explore, and analyze your dbt projects and data models with intelligent search and question-answering capabilities.

## What I Can Do

### 🔍 **Model Discovery & Search**
- **List Models**: Browse all your dbt models with filtering by project, schema, or materialization type
- **Semantic Search**: Find relevant models using natural language queries (e.g., "customer revenue models" or "user behavior data")
- **Model Details**: Get comprehensive information about specific models including SQL, documentation, and lineage

### 🤔 **Intelligent Question Answering**
- Ask analytical questions about your data in plain English
- Get insights backed by relevant dbt models and supporting SQL queries
- Understand relationships between different data models

### 📊 **Project Overview**
- Get summaries of your connected dbt projects
- Understand model relationships and dependencies
- Explore different materialization strategies

## My Tools Available to You

1. **`list_dbt_models`** - Browse and filter your dbt models
2. **`search_dbt_models`** - Find models using natural language search
3. **`get_model_details`** - Deep dive into specific models with SQL and lineage
4. **`get_project_summary`** - Get overview of your dbt projects

## How to Get Started

1. **Explore your models**: Start with `get_project_summary` to understand what data you have
2. **Search for relevant data**: Use `search_dbt_models` to find models related to your analysis
3. **Dive deeper**: Use `get_model_details` to understand specific models better
4. **Compose your analysis**: Use the discovered models and their SQL to build your insights

## Tips for Best Results

- **Be specific** in your search queries (e.g., "customer acquisition models" vs "customer data")
- **Provide context** when asking questions to get more relevant answers
- **Explore dependencies** to understand how models relate to each other
- **Ask follow-up questions** to dive deeper into interesting findings

Ready to explore your data? Start with a project summary or search for relevant models to build your analysis!"""

    return {
        "content": [{"type": "text", "text": content}],
        "description": "Introduction to Ragstar capabilities and available tools",
    }


def _get_data_analysis_guidance() -> Dict[str, Any]:
    """Guidance on effective data analysis with Ragstar."""
    content = """# Data Analysis with Ragstar 📈

## Asking Effective Questions

### 🎯 **Be Specific and Contextual**
Instead of: *"Show me sales data"*
Try: *"What are the top 5 products by revenue in the last quarter, and how do they compare to the previous quarter?"*

### 🔗 **Build on Previous Insights**
- Start with broad questions, then drill down
- Reference specific models or metrics you've discovered
- Ask about relationships between different data points

### 💡 **Great Question Examples**

**Business Metrics:**
- "What is our customer acquisition cost trend over the past 6 months?"
- "Which marketing channels are driving the highest lifetime value customers?"
- "How has our monthly recurring revenue grown year-over-year?"

**Operational Insights:**
- "What percentage of orders are shipped within 2 days?"
- "Which product categories have the highest return rates?"
- "How does customer support response time correlate with satisfaction scores?"

**Cohort & Segmentation:**
- "How do user retention rates differ between subscription tiers?"
- "What are the characteristics of our most valuable customer segment?"
- "How has user engagement changed since the new feature launch?"

## Analysis Workflow

### 1. **Discovery Phase**
Start with: get_project_summary
→ Understand what data domains you have
→ Identify key business areas

### 2. **Exploration Phase**  
Use: search_dbt_models with business terms
→ Find models related to your analysis area
→ Review model descriptions and metadata

### 3. **Investigation Phase**
Use: get_model_details on relevant models
→ Examine SQL and business logic
→ Understand data transformations and calculations

### 4. **Synthesis Phase**
Combine discovered models and their SQL
→ Build comprehensive understanding of data flow
→ Compose analytical insights from multiple sources

Ready to start your analysis? Begin by exploring your project structure and discovering relevant models!"""

    return {
        "content": [{"type": "text", "text": content}],
        "description": "Guidance on effective data analysis approaches",
    }


def _get_model_exploration_guide(focus_area: Optional[str] = None) -> Dict[str, Any]:
    """Guide for exploring dbt models."""
    content = """# Exploring dbt Models with Ragstar 🗺️

## Model Discovery Strategies

### 🔎 **Finding Relevant Models**
1. **Browse by Category**: Use `list_dbt_models` with filters
2. **Semantic Search**: Use `search_dbt_models` with business terms
3. **Project Overview**: Start with `get_project_summary`

### 📋 **Understanding Model Metadata**
- **Materialization**: Table, view, incremental, ephemeral
- **Tags**: Business domains, data quality, frequency
- **Schema**: Logical grouping and organization
- **Dependencies**: Upstream and downstream relationships

## Deep Diving into Models

### 🔍 **Model Details Investigation**
Use `get_model_details` to get:
- Raw and compiled SQL
- Column definitions and data types
- Business logic and transformations
- Dependencies and lineage

### 🕸️ **Understanding Data Lineage**
- **Upstream**: What models/sources feed into this model
- **Downstream**: What models depend on this model
- **Impact Analysis**: How changes affect downstream consumers

## Best Practices for Model Exploration

### 🎯 **Focused Exploration**
- Start with high-level business questions
- Follow the data lineage to understand calculations
- Look for documentation and business context

### 🔄 **Iterative Discovery**
- Use search results to discover related models
- Follow interesting dependencies
- Ask questions about unexpected relationships

Ready to explore? Try searching for models related to your current analysis!"""

    return {
        "content": [{"type": "text", "text": content}],
        "description": "Guide for exploring and understanding dbt models",
    }


def _get_best_practices(topic: Optional[str] = None) -> Dict[str, Any]:
    """Best practices for working with Ragstar and dbt."""
    content = """# Ragstar Best Practices 🎯

## Effective Question Asking

### 📝 **Question Structure**
- Start with business context
- Be specific about time periods
- Mention relevant metrics or KPIs
- Ask for comparisons when relevant

### 🔍 **Search Strategies**
- Use business terms, not technical jargon
- Include domain-specific keywords
- Try different phrasings if first search doesn't work
- Look for model descriptions and tags

## Working with Results

### 📊 **Interpreting Outputs**
- Review the SQL to understand calculations
- Check model dependencies for data sources
- Validate results against known benchmarks
- Ask follow-up questions for clarification

### 🔄 **Iterative Analysis**
- Build on previous findings
- Explore interesting anomalies
- Cross-reference different data sources
- Document key insights for future reference

## Collaboration Tips

### 👥 **Sharing Insights**
- Document interesting findings
- Share relevant SQL queries with team
- Reference specific models in discussions
- Build common understanding of metrics

Ready to put these practices to work? Start with a specific question about your data!"""

    return {
        "content": [{"type": "text", "text": content}],
        "description": "Best practices for effective data analysis with Ragstar",
    }
