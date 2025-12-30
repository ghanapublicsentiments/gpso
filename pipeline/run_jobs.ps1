# PowerShell script for pipeline jobs
# Test runner for pipeline jobs
# Usage: .\run_jobs.ps1 [run_id]

param(
    [string]$RunId
)

$ErrorActionPreference = "Stop"  # Exit on error

Write-Host "════════════════════════════════════════════════════════════"
Write-Host "  GPSO Pipeline Job Test Runner"
Write-Host "════════════════════════════════════════════════════════════"
Write-Host ""

if ($RunId) {
    Write-Host "Mode: CONTINUE from Run ID: $RunId"
    Write-Host ""
    
    # Continue mode - skip Job 0
    Write-Host "⊘ Skipping Job 0 (continue mode)"
    Write-Host ""
    
    $PIPELINE_RUN_ID = $RunId
} else {
    Write-Host "Mode: NEW RUN"
    Write-Host ""
    
    # Job 0: Initialize Pipeline
    Write-Host "════════════════════════════════════════════════════════════"
    Write-Host "  Job 0: Initialize Pipeline"
    Write-Host "════════════════════════════════════════════════════════════"
    
    $tempLog = "$env:TEMP\gpso_job0.log"
    uv run python jobs/job0_init_pipeline.py | Tee-Object -FilePath $tempLog
    
    # Extract Pipeline Run ID
    $logContent = Get-Content $tempLog -Raw
    if ($logContent -match 'Pipeline Run ID: (\d+)') {
        $PIPELINE_RUN_ID = $matches[1]
    } else {
        Write-Host ""
        Write-Host "❌ ERROR: Could not extract Pipeline Run ID from Job 0 output"
        exit 1
    }
    
    Write-Host ""
    Write-Host "✓ Job 0 Complete - Pipeline Run ID: $PIPELINE_RUN_ID"
    Write-Host ""
    
    # Job 1: Data Collection
    Write-Host "════════════════════════════════════════════════════════════"
    Write-Host "  Job 1: Data Collection"
    Write-Host "════════════════════════════════════════════════════════════"
    uv run python jobs/job1_data_collection.py --run-id $PIPELINE_RUN_ID
    
    Write-Host ""
    Write-Host "✓ Job 1 Complete"
    Write-Host ""
}

# Job 2: Captions & Entities
Write-Host "════════════════════════════════════════════════════════════"
Write-Host "  Job 2: Captions & Entities"
Write-Host "════════════════════════════════════════════════════════════"
uv run python jobs/job2_captions_entities.py --run-id $PIPELINE_RUN_ID

Write-Host ""
Write-Host "✓ Job 2 Complete"
Write-Host ""

# Job 3: Sentiment Analysis
Write-Host "════════════════════════════════════════════════════════════"
Write-Host "  Job 3: Sentiment Analysis (EXPENSIVE)"
Write-Host "════════════════════════════════════════════════════════════"
Write-Host "⚠️  This job uses significant API tokens. Press Ctrl+C to cancel."
Start-Sleep -Seconds 5
uv run python jobs/job3_sentiment_analysis.py --run-id $PIPELINE_RUN_ID

Write-Host ""
Write-Host "✓ Job 3 Complete"
Write-Host ""

# Job 4: Post Processing
Write-Host "════════════════════════════════════════════════════════════"
Write-Host "  Job 4: Post Processing"
Write-Host "════════════════════════════════════════════════════════════"
uv run python jobs/job4_post_processing.py --run-id $PIPELINE_RUN_ID

Write-Host ""
Write-Host "✓ Job 4 Complete"
Write-Host ""

# Job 5: Complete Pipeline
Write-Host "════════════════════════════════════════════════════════════"
Write-Host "  Job 5: Complete Pipeline"
Write-Host "════════════════════════════════════════════════════════════"
uv run python jobs/job5_complete_pipeline.py --run-id $PIPELINE_RUN_ID

Write-Host ""
Write-Host "✓ Job 5 Complete"
Write-Host ""

# Summary
Write-Host "════════════════════════════════════════════════════════════"
Write-Host "  ✅ FULL PIPELINE COMPLETE"
Write-Host "════════════════════════════════════════════════════════════"
Write-Host "Pipeline Run ID: $PIPELINE_RUN_ID"
Write-Host ""
Write-Host "View results in BigQuery:"
Write-Host "  • Sentiments: pipeline_comment_sentiments"
Write-Host "  • Summaries: pipeline_entity_summaries"
Write-Host "  • Checkpoints: pipeline_checkpoints"
Write-Host ""
