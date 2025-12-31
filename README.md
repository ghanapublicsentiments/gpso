# Ghana Public Sentiments Observatory (GPSO)

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![BigQuery](https://img.shields.io/badge/database-BigQuery-blue.svg)](https://cloud.google.com/bigquery)

**Ghana Public Sentiments Observatory (GPSO)** is an automated sentiment analysis platform that monitors and analyzes public opinion in Ghana by processing social media comments from YouTube and Facebook, extracting entity-level sentiments using LLMs.

---

## 📋 Table of Contents

- [Overview & Features](#-overview--features)
- [Setup](#-setup)
- [Configuration](#-configuration)
- [Running the Pipeline](#-running-the-pipeline)
- [Running the Streamlit App](#-running-the-streamlit-app)
- [Querying BigQuery Datasets](#-querying-bigquery-datasets)
- [Contributing](#-contributing)

---

## 🔍 Overview & Features

GPSO provides automated sentiment analysis by:

1. **Collecting** YouTube video comments and Facebook public page posts via official APIs
2. **Processing** comments through entity detection, sentiment analysis, and summarization using LLMs
3. **Storing** processed data in Google BigQuery
4. **Visualizing** insights through an interactive Streamlit dashboard

### Key Features

- **Multi-Platform Data Collection**: Scrapes content from YouTube channels (via YouTube Data API v3) and Facebook public pages (via Facebook Graph API)
- **Entity-Level Sentiment Analysis**: Analyzes sentiments toward specific people, organizations, and issues
- **Multi-LLM Support**: Compatible with OpenAI, Anthropic, Google Gemini, Grok, and NVIDIA models
- **KNN Sentiment Smoothing**: Reduces noise in sentiment signals using k-nearest neighbors
- **Channel Normalization**: Normalizes sentiments across different sources using historical ECDFs
- **BigQuery Storage**: Scalable cloud data warehouse with public read access to processed datasets
- **Interactive Dashboard**: Real-time visualizations, source filtering, AI chat interface, and methodology documentation

---

## 🔧 Setup

### Prerequisites

- **Python 3.13+**
- **API Keys** (for running pipeline):
  - At least one LLM provider: OpenAI, Anthropic, Google (Gemini), Grok, or NVIDIA
  - Google API key for YouTube Data API v3

### Installation

**Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows PowerShell:**
```powershell
.\setup.ps1
```

**Windows CMD:**
```cmd
setup.bat
```

The setup script will:
1. Install [uv](https://docs.astral.sh/uv/) package manager
2. Create a virtual environment
3. Install all dependencies
4. Install Playwright browsers

---

## ⚙️ Configuration

### 1. Environment Variables (`.env`)

Copy `.env.sample` to `.env` and add your API keys:

```bash
cp .env.sample .env
```

Edit `.env`:
```bash
# LLM API Keys (choose at least one provider)
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-gemini-api-key-here
GROK_API_KEY=your-grok-api-key-here
NVIDIA_API_KEY=your-nvidia-api-key-here

# Facebook Graph API
FACEBOOK_ACCESS_TOKEN=your-facebook-access-token-here

# BigQuery (if you want to use your own database)
# Place your service account JSON key as: bigquery-credentials.json
# BIGQUERY_DATASET=gpso_main
```

### 2. Pipeline Configuration (`pipeline/pipeline_config.py`)

Pipeline behavior can be customized by modifying the configuration dataclasses in `pipeline/pipeline_config.py`. Available settings include:

- **Data Collection**: YouTube video limits, comment thresholds
- **Caption Generation**: LLM model selection, similarity thresholds, embedding models
- **Sentiment Analysis**: Model selection, worker concurrency, rate limiting
- **Smoothing**: KNN parameters for noise reduction
- **Normalization**: Historical window size, normalization strength
- **Summaries**: Model selection, concurrency settings

Modify these configurations before running the pipeline to customize behavior for your specific needs.

### 3. Available LLM Models

The platform supports multiple LLM providers via OpenAI-compatible APIs (configured in `config.py`):

**OpenAI Models:**
- `gpt-5-nano` (default - fast & cost-effective)
- `gpt-5-mini`
- `gpt-5.1`
- `gpt-5.2`

**Google Gemini Models:**
- `gemini-2.5-flash-lite`
- `gemini-3-flash-preview`
- `gemini-3-pro-preview`

**Anthropic Claude Models:**
- `claude-haiku-4-5`
- `claude-sonnet-4-5`
- `claude-opus-4-5`

**Grok Models:**
- `grok-4-1-fast-reasoning`
- `grok-4-1-fast-non-reasoning`

**NVIDIA Models:**
- `openai/gpt-oss-20b`

---

## 🚀 Running the Pipeline

The pipeline processes YouTube and Facebook data through multiple stages. You can run individual jobs or the complete pipeline.

### Testing Data Collection (Optional)

Before running the full pipeline, you can test the data collection independently:

**Test Facebook scraper:**
```bash
# Test all configured Facebook pages
python test_facebook_scraper.py

# Test a specific page
python test_facebook_scraper.py --page "MyJoyOnline"

# Adjust limits
python test_facebook_scraper.py --max-posts 5 --max-comments 20
```

This verifies your Facebook API credentials and page configurations without touching the database.

### Option 1: Run Complete Pipeline (Recommended)

**Linux/macOS:**
```bash
cd pipeline
./run_jobs.sh
```

**Windows PowerShell:**
```powershell
cd pipeline
.\run_jobs.ps1
```

**Windows CMD:**
```cmd
cd pipeline
run_jobs.bat
```

This runs all jobs in sequence:
- **Job 0**: Initialize pipeline run (creates run ID)
- **Job 1**: Collect YouTube videos and Facebook posts
- **Job 2**: Generate captions and detect entities
- **Job 3**: Analyze sentiments
- **Job 4**: Apply smoothing and normalization, generate summaries

### Option 2: Continue from Existing Run

If a pipeline run was interrupted, continue from where it left off:

```bash
# Linux/macOS
./run_jobs.sh <run_id>

# Windows PowerShell
.\run_jobs.ps1 <run_id>

# Windows CMD
run_jobs.bat <run_id>
```

### Option 3: Run Individual Jobs

```bash
# Job 0: Initialize pipeline
uv run python jobs/job0_init_pipeline.py

# Job 1: Data collection (requires run ID from Job 0)
uv run python jobs/job1_data_collection.py --run-id <run_id>

# Job 2: Captions & entity detection
uv run python jobs/job2_captions_entities.py --run-id <run_id>

# Job 3: Sentiment analysis
uv run python jobs/job3_sentiment_analysis.py --run-id <run_id>

# Job 4: Post-processing (smoothing, normalization, summaries)
uv run python jobs/job4_post_processing.py --run-id <run_id>

# Job 5: Complete pipeline (all stages)
uv run python jobs/job5_complete_pipeline.py --run-id <run_id>
```

### Pipeline Output

The pipeline stores data in BigQuery tables:
- **Raw data**: `youtube_videos`, `youtube_comments`, `facebook_posts`, `facebook_comments`
- **Processed data**: `pipeline_comment_sentiments`, `pipeline_entity_summaries`
- **Metadata**: `pipeline_runs`

---

## 🎨 Running the Streamlit App

The Streamlit dashboard provides an interactive interface to explore sentiment data.

### Start the App

```bash
uv run streamlit run app/main.py
```

The app opens at `http://localhost:8501`.

### Features

- **Home**: Real-time sentiment scores, trends, and entity cards
- **Chat**: AI-powered natural language queries about sentiment data
- **Methodology**: Detailed pipeline documentation
- **FAQs**: Common questions and usage guidelines
- **Playground**: Interactive model experimentation

---

## 🗃 Querying BigQuery Datasets

**Note:** Only the `pipeline_entity_summaries` table is publicly available for querying. This table contains aggregated sentiment data used by the Streamlit dashboard.

### Dataset Information

- **Project ID**: `gpso-478602`
- **Dataset**: `gpso_main`
- **Public Table**: `pipeline_entity_summaries`

### Query Examples

#### Get Latest Sentiment Summaries

```sql
SELECT 
    entity_name,
    avg_sentiment,
    sentiment_count,
    content_count,
    sentiment_summary
FROM `gpso-478602.gpso_main.pipeline_entity_summaries`
WHERE run_id = (
    SELECT MAX(id) 
    FROM `gpso-478602.gpso_main.pipeline_runs` 
    WHERE status = 'completed'
)
ORDER BY sentiment_count DESC
LIMIT 10;
```

#### Get Sentiment Trends (Last 30 Days)

```sql
SELECT 
    DATE(pr.run_date) as date,
    pes.entity_name,
    pes.avg_sentiment,
    pes.sentiment_count
FROM `gpso-478602.gpso_main.pipeline_entity_summaries` pes
JOIN `gpso-478602.gpso_main.pipeline_runs` pr ON pes.run_id = pr.id
WHERE pr.status = 'completed'
    AND pr.run_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
ORDER BY date DESC, entity_name;
```

#### Most Discussed Topics (Last 7 Days)

```sql
SELECT 
    entity_name,
    SUM(sentiment_count) as total_mentions,
    AVG(avg_sentiment) as avg_sentiment_score
FROM `gpso-478602.gpso_main.pipeline_entity_summaries`
WHERE run_id IN (
    SELECT id 
    FROM `gpso-478602.gpso_main.pipeline_runs` 
    WHERE status = 'completed' 
    AND run_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
)
GROUP BY entity_name
ORDER BY total_mentions DESC
LIMIT 20;
```

### Access Methods

**BigQuery Console (Web):**
1. Go to [BigQuery Console](https://console.cloud.google.com/bigquery)
2. Navigate to `gpso-478602.gpso_main.pipeline_entity_summaries`
3. Run your queries

**Python:**
```python
from google.cloud import bigquery

client = bigquery.Client()
query = """
SELECT entity_name, avg_sentiment, sentiment_count
FROM `gpso-478602.gpso_main.pipeline_entity_summaries`
ORDER BY sentiment_count DESC LIMIT 10
"""
df = client.query(query).to_dataframe()
```

**CLI:**
```bash
bq query --use_legacy_sql=false '
SELECT entity_name, avg_sentiment, sentiment_count
FROM `gpso-478602.gpso_main.pipeline_entity_summaries`
ORDER BY sentiment_count DESC LIMIT 10'
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

### 1. Fork the Repository

```bash
git clone https://github.com/ghanapublicsentiments/gpso.git
cd gpso
git checkout -b feature/your-feature-name
```

### 2. Set Up Development Environment

```bash
./setup.sh  # or setup.ps1 on Windows
```

### 3. Make Changes

- Add features or fix bugs
- Update tests if applicable
- Follow existing code style and conventions

### 4. Test Your Changes

```bash
# Run the pipeline
cd pipeline
./run_jobs.sh

# Test the Streamlit app
uv run streamlit run app/main.py
```

### 5. Submit Pull Request

- Push to your fork
- Create a pull request with a clear description
- Reference any related issues

### Development Guidelines

- **Code Style**: Follow PEP 8 for Python code
- **Documentation**: Update README and docstrings for new features
- **Testing**: Ensure existing functionality isn't broken
- **Commits**: Write clear, descriptive commit messages

### Areas for Contribution

- **Data Sources**: Add new YouTube channels or Facebook pages data sources
- **LLM Models**: Integrate additional LLM providers
- **Visualizations**: Enhance dashboard charts and UI
- **Performance**: Optimize pipeline stages
- **Documentation**: Improve guides and examples

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

