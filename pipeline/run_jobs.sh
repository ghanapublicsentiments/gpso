#!/usr/bin/env bash
# Test runner for pipeline jobs
# Usage: ./run_jobs.sh [run_id]

set -e  # Exit on error

RUN_ID=$1

echo "════════════════════════════════════════════════════════════"
echo "  GPSO Pipeline Job Test Runner"
echo "════════════════════════════════════════════════════════════"
echo ""

if [ -n "$RUN_ID" ]; then
    echo "Mode: CONTINUE from Run ID: $RUN_ID"
    echo ""
    
    # Continue mode - skip Job 0
    echo "⊘ Skipping Job 0 (continue mode)"
    echo ""
    
    PIPELINE_RUN_ID=$RUN_ID
else
    echo "Mode: NEW RUN"
    echo ""
    
    # Job 0: Initialize Pipeline
    echo "════════════════════════════════════════════════════════════"
    echo "  Job 0: Initialize Pipeline"
    echo "════════════════════════════════════════════════════════════"
    uv run python jobs/job0_init_pipeline.py | tee /tmp/gpso_job0.log
    
    # Extract Pipeline Run ID (macOS compatible)
    PIPELINE_RUN_ID=$(grep -o 'Pipeline Run ID: [0-9]*' /tmp/gpso_job0.log | grep -o '[0-9]*' | tail -1)
    
    if [ -z "$PIPELINE_RUN_ID" ]; then
        echo ""
        echo "❌ ERROR: Could not extract Pipeline Run ID from Job 0 output"
        exit 1
    fi
    
    echo ""
    echo "✓ Job 0 Complete - Pipeline Run ID: $PIPELINE_RUN_ID"
    echo ""
    
    # Job 1: Data Collection
    echo "════════════════════════════════════════════════════════════"
    echo "  Job 1: Data Collection"
    echo "════════════════════════════════════════════════════════════"
    uv run python jobs/job1_data_collection.py --run-id $PIPELINE_RUN_ID
    
    echo ""
    echo "✓ Job 1 Complete"
    echo ""
fi

# Job 2: Captions & Entities
echo "════════════════════════════════════════════════════════════"
echo "  Job 2: Captions & Entities"
echo "════════════════════════════════════════════════════════════"
uv run python jobs/job2_captions_entities.py --run-id $PIPELINE_RUN_ID

echo ""
echo "✓ Job 2 Complete"
echo ""

# Job 3: Sentiment Analysis
echo "════════════════════════════════════════════════════════════"
echo "  Job 3: Sentiment Analysis (EXPENSIVE)"
echo "════════════════════════════════════════════════════════════"
echo "⚠️  This job uses significant API tokens. Press Ctrl+C to cancel."
sleep 5
uv run python jobs/job3_sentiment_analysis.py --run-id $PIPELINE_RUN_ID

echo ""
echo "✓ Job 3 Complete"
echo ""

# Job 4: Post Processing
echo "════════════════════════════════════════════════════════════"
echo "  Job 4: Post Processing"
echo "════════════════════════════════════════════════════════════"
uv run python jobs/job4_post_processing.py --run-id $PIPELINE_RUN_ID

echo ""
echo "✓ Job 4 Complete"
echo ""

# Job 5: Complete Pipeline
echo "════════════════════════════════════════════════════════════"
echo "  Job 5: Complete Pipeline"
echo "════════════════════════════════════════════════════════════"
uv run python jobs/job5_complete_pipeline.py --run-id $PIPELINE_RUN_ID

echo ""
echo "✓ Job 5 Complete"
echo ""

# Summary
echo "════════════════════════════════════════════════════════════"
echo "  ✅ FULL PIPELINE COMPLETE"
echo "════════════════════════════════════════════════════════════"
echo "Pipeline Run ID: $PIPELINE_RUN_ID"
echo ""
echo "View results in BigQuery:"
echo "  • Sentiments: pipeline_comment_sentiments"
echo "  • Summaries: pipeline_entity_summaries"
echo "  • Checkpoints: pipeline_checkpoints"
echo ""
