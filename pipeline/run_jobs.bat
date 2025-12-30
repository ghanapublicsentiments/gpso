@echo off
REM Batch script for pipeline jobs
REM Test runner for pipeline jobs
REM Usage: run_jobs.bat [run_id]

setlocal enabledelayedexpansion

set RUN_ID=%1

echo ════════════════════════════════════════════════════════════
echo   GPSO Pipeline Job Test Runner
echo ════════════════════════════════════════════════════════════
echo.

if not "%RUN_ID%"=="" (
    echo Mode: CONTINUE from Run ID: %RUN_ID%
    echo.
    
    REM Continue mode - skip Job 0
    echo ⊘ Skipping Job 0 (continue mode^)
    echo.
    
    set PIPELINE_RUN_ID=%RUN_ID%
) else (
    echo Mode: NEW RUN
    echo.
    
    REM Job 0: Initialize Pipeline
    echo ════════════════════════════════════════════════════════════
    echo   Job 0: Initialize Pipeline
    echo ════════════════════════════════════════════════════════════
    
    set "TEMP_LOG=%TEMP%\gpso_job0.log"
    uv run python jobs/job0_init_pipeline.py > "!TEMP_LOG!" 2>&1
    if errorlevel 1 (
        echo.
        echo ❌ ERROR: Job 0 failed
        exit /b 1
    )
    
    REM Display log content
    type "!TEMP_LOG!"
    
    REM Extract Pipeline Run ID
    for /f "tokens=4" %%i in ('findstr /C:"Pipeline Run ID:" "!TEMP_LOG!"') do set PIPELINE_RUN_ID=%%i
    
    if "!PIPELINE_RUN_ID!"=="" (
        echo.
        echo ❌ ERROR: Could not extract Pipeline Run ID from Job 0 output
        exit /b 1
    )
    
    echo.
    echo ✓ Job 0 Complete - Pipeline Run ID: !PIPELINE_RUN_ID!
    echo.
    
    REM Job 1: Data Collection
    echo ════════════════════════════════════════════════════════════
    echo   Job 1: Data Collection
    echo ════════════════════════════════════════════════════════════
    uv run python jobs/job1_data_collection.py --run-id !PIPELINE_RUN_ID!
    if errorlevel 1 (
        echo.
        echo ❌ ERROR: Job 1 failed
        exit /b 1
    )
    
    echo.
    echo ✓ Job 1 Complete
    echo.
)

REM Job 2: Captions ^& Entities
echo ════════════════════════════════════════════════════════════
echo   Job 2: Captions ^& Entities
echo ════════════════════════════════════════════════════════════
uv run python jobs/job2_captions_entities.py --run-id %PIPELINE_RUN_ID%
if errorlevel 1 (
    echo.
    echo ❌ ERROR: Job 2 failed
    exit /b 1
)

echo.
echo ✓ Job 2 Complete
echo.

REM Job 3: Sentiment Analysis
echo ════════════════════════════════════════════════════════════
echo   Job 3: Sentiment Analysis (EXPENSIVE^)
echo ════════════════════════════════════════════════════════════
echo ⚠️  This job uses significant API tokens. Press Ctrl+C to cancel.
timeout /t 5 /nobreak >nul
uv run python jobs/job3_sentiment_analysis.py --run-id %PIPELINE_RUN_ID%
if errorlevel 1 (
    echo.
    echo ❌ ERROR: Job 3 failed
    exit /b 1
)

echo.
echo ✓ Job 3 Complete
echo.

REM Job 4: Post Processing
echo ════════════════════════════════════════════════════════════
echo   Job 4: Post Processing
echo ════════════════════════════════════════════════════════════
uv run python jobs/job4_post_processing.py --run-id %PIPELINE_RUN_ID%
if errorlevel 1 (
    echo.
    echo ❌ ERROR: Job 4 failed
    exit /b 1
)

echo.
echo ✓ Job 4 Complete
echo.

REM Job 5: Complete Pipeline
echo ════════════════════════════════════════════════════════════
echo   Job 5: Complete Pipeline
echo ════════════════════════════════════════════════════════════
uv run python jobs/job5_complete_pipeline.py --run-id %PIPELINE_RUN_ID%
if errorlevel 1 (
    echo.
    echo ❌ ERROR: Job 5 failed
    exit /b 1
)

echo.
echo ✓ Job 5 Complete
echo.

REM Summary
echo ════════════════════════════════════════════════════════════
echo   ✅ FULL PIPELINE COMPLETE
echo ════════════════════════════════════════════════════════════
echo Pipeline Run ID: %PIPELINE_RUN_ID%
echo.
echo View results in BigQuery:
echo   • Sentiments: pipeline_comment_sentiments
echo   • Summaries: pipeline_entity_summaries
echo   • Checkpoints: pipeline_checkpoints
echo.

endlocal
