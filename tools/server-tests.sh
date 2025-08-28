#!/bin/bash
#
# 🧠 MLX Embedding & Reranking Comprehensive Test Suite
#
# Automated testing framework for embedding quality, reranking performance, 
# and system benchmarks using your configured MLX model.
#
# Features:
# - 🏥 Server health and configuration validation
# - 🔤 Embedding quality assessment (semantic similarity, multilingual)
# - 🔄 Reranking functionality and accuracy testing  
# - ⚡ Performance benchmarking (latency, throughput, stress testing)
# - 💾 Result storage and reporting
#
# Usage:
#     ./tools/server-tests.sh                    # Full test suite
#     ./tools/server-tests.sh --quick            # Quick validation only
#     ./tools/server-tests.sh --performance      # Performance tests only
#     ./tools/server-tests.sh --url localhost:8080  # Custom server URL
#

set -e  # Exit on any error

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Default configuration
SERVER_URL="http://localhost:9000"
TEST_MODE="full"  # full, quick, performance, quality
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_PREFIX="test_${TIMESTAMP}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables from .env if it exists
ENV_FILE="$PROJECT_ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
    # Source .env file (but only export specific variables we care about)
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ $key =~ ^[[:space:]]*# ]] && continue
        [[ -z $key ]] && continue
        # Remove quotes from value
        value=$(echo "$value" | sed 's/^["'\'']\|["'\'']$//g')
        # Export relevant variables
        case $key in
            PORT) ENV_PORT="$value" ;;
            HOST) ENV_HOST="$value" ;;
        esac
    done < "$ENV_FILE"
    
    # Update SERVER_URL if PORT or HOST found in .env
    if [[ -n "$ENV_HOST" && -n "$ENV_PORT" ]]; then
        SERVER_URL="http://${ENV_HOST}:${ENV_PORT}"
    elif [[ -n "$ENV_PORT" ]]; then
        SERVER_URL="http://localhost:${ENV_PORT}"
    fi
fi

# Ensure output path is inside the tools directory next to this script
OUTPUT_DIR="$SCRIPT_DIR/test-results"

print_header() {
    echo -e "\n${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${WHITE}$1${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
}

print_status() {
    local status="$1"
    local message="$2"
    if [[ "$status" == "success" ]]; then
        echo -e "${GREEN}✅ $message${NC}"
    elif [[ "$status" == "error" ]]; then
        echo -e "${RED}❌ $message${NC}"
    elif [[ "$status" == "warning" ]]; then
        echo -e "${YELLOW}⚠️  $message${NC}"
    else
        echo -e "${BLUE}ℹ️  $message${NC}"
    fi
}

print_step() {
    echo -e "${BLUE}🔄 $1${NC}"
}

check_prerequisites() {
    print_header "🔍 Prerequisites Check"
    
    # Check if Python 3 is available
    if ! command -v python3 &> /dev/null; then
        print_status "error" "Python 3 not found. Please install Python 3."
        exit 1
    fi
    print_status "success" "Python 3 found: $(python3 --version)"
    
    # Check if required Python packages are available
    print_step "Checking Python dependencies..."
    
    local missing_packages=()
    
    # Check for requests
    if ! python3 -c "import requests" 2>/dev/null; then
        missing_packages+=("requests")
    fi
    
    # Check for other basic packages
    if ! python3 -c "import json, time, statistics, math" 2>/dev/null; then
        print_status "error" "Basic Python packages missing"
        exit 1
    fi
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_status "warning" "Missing packages: ${missing_packages[*]}"
        print_step "Installing missing packages..."
        
        # Try different installation methods
        if [[ -d "$PROJECT_ROOT/.venv" ]]; then
            # Use virtual environment if available
            source "$PROJECT_ROOT/.venv/bin/activate"
            pip install "${missing_packages[@]}" || {
                print_status "error" "Failed to install required packages in virtual environment"
                exit 1
            }
        elif command -v pip3 &> /dev/null; then
            # Try with --user flag first
            pip3 install --user "${missing_packages[@]}" 2>/dev/null || {
                # If that fails, try with --break-system-packages
                print_status "warning" "Installing packages with --break-system-packages flag"
                pip3 install --break-system-packages "${missing_packages[@]}" || {
                    print_status "error" "Failed to install required packages"
                    print_status "info" "Please install manually: pip3 install ${missing_packages[*]}"
                    print_status "info" "Or create a virtual environment: python3 -m venv .venv && source .venv/bin/activate"
                    exit 1
                }
            }
        else
            print_status "error" "pip3 not found"
            exit 1
        fi
    fi
    
    print_status "success" "All Python dependencies available"
    
    # Check if test scripts exist
    if [[ ! -f "$SCRIPT_DIR/tests/validate-quality-simple.py" ]]; then
        print_status "error" "Quality validation script not found: $SCRIPT_DIR/tests/validate-quality-simple.py"
        exit 1
    fi
    
    if [[ ! -f "$SCRIPT_DIR/tests/benchmark-performance.py" ]]; then
        print_status "error" "Performance benchmark script not found: $SCRIPT_DIR/tests/benchmark-performance.py"
        exit 1
    fi
    
    print_status "success" "All test scripts found"
}

check_server_connectivity() {
    print_header "🏥 Server Connectivity Check"
    
    print_step "Testing connection to $SERVER_URL..."
    
    # Try to connect to health endpoint
    if curl -s --max-time 10 "$SERVER_URL/health/" > /dev/null 2>&1; then
        print_status "success" "Server is responding at $SERVER_URL"
        
        # Get server info
        local server_info=$(curl -s --max-time 10 "$SERVER_URL/health/")
        local backend=$(echo "$server_info" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('backend', {}).get('name', 'Unknown'))" 2>/dev/null || echo "Unknown")
        local model=$(echo "$server_info" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('backend', {}).get('model_name', 'Unknown'))" 2>/dev/null || echo "Unknown")
        
        print_status "info" "Backend: $backend"
        print_status "info" "Model: $model"
        
        return 0
    else
        print_status "error" "Cannot connect to server at $SERVER_URL"
        print_status "info" "Please ensure your server is running:"
        print_status "info" "  ./tools/server-run.sh"
        print_status "info" "Or check if it's running on a different port"
        return 1
    fi
}

create_output_directory() {
    print_step "Creating output directory: $OUTPUT_DIR"
    
    mkdir -p "$OUTPUT_DIR"
    
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        print_status "error" "Failed to create output directory: $OUTPUT_DIR"
        exit 1
    fi
    
    print_status "success" "Output directory ready: $OUTPUT_DIR"
}

run_quality_validation() {
    print_header "🧠 Running Quality Validation Tests"
    
    local output_file="$OUTPUT_DIR/${RESULTS_PREFIX}_quality_validation.json"
    local log_file="$OUTPUT_DIR/${RESULTS_PREFIX}_quality_validation.log"
    
    print_step "Running comprehensive quality validation..."
    print_status "info" "Output: $output_file"
    print_status "info" "Log: $log_file"
    
    # Run quality validation with both output formats
    if python3 "$SCRIPT_DIR/tests/validate-quality-simple.py" \
        --url "$SERVER_URL" \
        --output "$output_file" \
        2>&1 | tee "$log_file"; then
        
        print_status "success" "Quality validation completed successfully"
        
        # Extract summary from JSON if available
        if [[ -f "$output_file" ]]; then
            local summary=$(python3 -c "
import json, sys
try:
    with open('$output_file', 'r') as f:
        data = json.load(f)
    summary = data.get('validation_summary', {})
    print(f\"Status: {summary.get('overall_status', 'Unknown')}\")
    print(f\"Success Rate: {summary.get('success_rate', 'Unknown')}\")
    print(f\"Time: {summary.get('total_validation_time', 'Unknown')}\")
except Exception as e:
    print(f\"Could not parse results: {e}\")
" 2>/dev/null)
            
            if [[ -n "$summary" ]]; then
                echo -e "${CYAN}📊 Quality Validation Summary:${NC}"
                echo "$summary" | while read line; do
                    print_status "info" "$line"
                done
            fi
        fi
        
        return 0
    else
        print_status "error" "Quality validation failed"
        return 1
    fi
}

run_performance_benchmark() {
    print_header "⚡ Running Performance Benchmark Tests"
    
    local output_file="$OUTPUT_DIR/${RESULTS_PREFIX}_performance_benchmark.json"
    local log_file="$OUTPUT_DIR/${RESULTS_PREFIX}_performance_benchmark.log"
    
    print_step "Running comprehensive performance benchmark..."
    print_status "info" "Output: $output_file"
    print_status "info" "Log: $log_file"
    
    # Run performance benchmark
    if python3 "$SCRIPT_DIR/tests/benchmark-performance.py" \
        --url "$SERVER_URL" \
        --output "$output_file" \
        --stress-duration 30 \
        2>&1 | tee "$log_file"; then
        
        print_status "success" "Performance benchmark completed successfully"
        
        # Extract summary from JSON if available
        if [[ -f "$output_file" ]]; then
            local summary=$(python3 -c "
import json, sys
try:
    with open('$output_file', 'r') as f:
        data = json.load(f)
    summary = data.get('benchmark_summary', {})
    print(f\"Status: {summary.get('overall_status', 'Unknown')}\")
    print(f\"Mean Latency: {summary.get('mean_latency_ms', 'Unknown')}ms\")
    print(f\"Peak Throughput: {summary.get('peak_throughput_texts_per_sec', 'Unknown')} texts/sec\")
    print(f\"Success Rate: {summary.get('stress_test_success_rate', 'Unknown')}\")
except Exception as e:
    print(f\"Could not parse results: {e}\")
" 2>/dev/null)
            
            if [[ -n "$summary" ]]; then
                echo -e "${CYAN}📊 Performance Benchmark Summary:${NC}"
                echo "$summary" | while read line; do
                    print_status "info" "$line"
                done
            fi
        fi
        
        return 0
    else
        print_status "error" "Performance benchmark failed"
        return 1
    fi
}

run_quick_validation() {
    print_header "🏃 Running Quick Validation"
    
    # Set up log files for quick validation
    local log_file="$OUTPUT_DIR/${RESULTS_PREFIX}_quick_validation.log"
    local summary_file="$OUTPUT_DIR/${RESULTS_PREFIX}_quick_validation.json"
    
    print_step "Running quick server validation..."
    print_status "info" "Log: $log_file"
    
    # Create log header
    {
        echo "=============================================="
        echo "🏃 Quick Validation Test - $(date)"
        echo "Server URL: $SERVER_URL"
        echo "=============================================="
        echo ""
    } > "$log_file"
    
    # Run the quality validator and capture both output and results
    if python3 "$SCRIPT_DIR/tests/validate-quality-simple.py" \
        --url "$SERVER_URL" \
        --output "$summary_file" \
        2>&1 | tee -a "$log_file"; then
        
        print_status "success" "Quick validation passed"
        
        # Add completion timestamp to log
        {
            echo ""
            echo "=============================================="
            echo "✅ Quick validation completed - $(date)"
            echo "=============================================="
        } >> "$log_file"
        
        return 0
    else
        print_status "error" "Quick validation failed"
        
        # Add failure timestamp to log
        {
            echo ""
            echo "=============================================="
            echo "❌ Quick validation failed - $(date)"
            echo "=============================================="
        } >> "$log_file"
        
        return 1
    fi
}

generate_comprehensive_report() {
    print_header "📊 Generating Comprehensive Report"
    
    local report_file="$OUTPUT_DIR/${RESULTS_PREFIX}_comprehensive_report.md"
    
    print_step "Creating comprehensive test report..."
    
    cat > "$report_file" << EOF
# 🧠 MLX Embedding & Reranking Test Report

**Generated:** $(date)
**Server URL:** $SERVER_URL
**Test Mode:** $TEST_MODE

## 📋 Test Summary

EOF

    # Add quality validation results if available
    local quality_file="$OUTPUT_DIR/${RESULTS_PREFIX}_quality_validation.json"
    if [[ -f "$quality_file" ]]; then
        echo "### 🧠 Quality Validation Results" >> "$report_file"
        echo "" >> "$report_file"
        
        python3 -c "
import json
try:
    with open('$quality_file', 'r') as f:
        data = json.load(f)
    
    summary = data.get('validation_summary', {})
    print(f\"- **Overall Status:** {summary.get('overall_status', 'Unknown')}\")
    print(f\"- **Success Rate:** {summary.get('success_rate', 'Unknown')}\")
    print(f\"- **Execution Time:** {summary.get('total_validation_time', 'Unknown')}\")
    print()
    
    # Server info
    health = data.get('server_health', {})
    if health.get('status') == 'healthy':
        print('### 🏥 Server Configuration')
        print()
        print(f\"- **Backend:** {health.get('backend', 'Unknown')}\")
        print(f\"- **Model:** {health.get('model', 'Unknown')}\")
        print(f\"- **Device:** {health.get('device', 'Unknown')}\")
        print()
    
    # Individual test results
    tests = ['basic_embedding', 'semantic_similarity', 'multilingual_support', 'reranking_quality']
    print('### 📝 Individual Test Results')
    print()
    for test in tests:
        test_data = data.get(test, {})
        status = test_data.get('status', 'unknown')
        emoji = '✅' if status == 'success' else '❌'
        test_name = test.replace('_', ' ').title()
        print(f\"- {emoji} **{test_name}:** {status}\")
    print()
        
except Exception as e:
    print(f'Error parsing quality results: {e}')
" >> "$report_file"
    fi

    # Add performance benchmark results if available
    local perf_file="$OUTPUT_DIR/${RESULTS_PREFIX}_performance_benchmark.json"
    if [[ -f "$perf_file" ]]; then
        echo "### ⚡ Performance Benchmark Results" >> "$report_file"
        echo "" >> "$report_file"
        
        python3 -c "
import json
try:
    with open('$perf_file', 'r') as f:
        data = json.load(f)
    
    summary = data.get('benchmark_summary', {})
    print(f\"- **Overall Status:** {summary.get('overall_status', 'Unknown')}\")
    print(f\"- **Mean Latency:** {summary.get('mean_latency_ms', 'Unknown')} ms\")
    print(f\"- **Peak Throughput:** {summary.get('peak_throughput_texts_per_sec', 'Unknown')} texts/sec\")
    print(f\"- **Stress Test Success Rate:** {summary.get('stress_test_success_rate', 'Unknown')}\")
    print()
    
    # Detailed latency results
    latency = data.get('embedding_latency', {})
    if latency:
        print('### 📊 Latency Details')
        print()
        print(f\"- **Mean:** {latency.get('mean_latency_ms', 'Unknown')} ms\")
        print(f\"- **P95:** {latency.get('p95_latency_ms', 'Unknown')} ms\")
        print(f\"- **P99:** {latency.get('p99_latency_ms', 'Unknown')} ms\")
        print()
        
except Exception as e:
    print(f'Error parsing performance results: {e}')
" >> "$report_file"
    fi

    # Add file locations
    echo "## 📁 Generated Files" >> "$report_file"
    echo "" >> "$report_file"
    
    for file in "$OUTPUT_DIR"/${RESULTS_PREFIX}_*; do
        if [[ -f "$file" ]]; then
            local filename=$(basename "$file")
            echo "- \`$filename\`" >> "$report_file"
        fi
    done
    
    echo "" >> "$report_file"
    echo "---" >> "$report_file"
    echo "*Report generated by MLX Embedding & Reranking Test Suite*" >> "$report_file"
    
    print_status "success" "Comprehensive report generated: $report_file"
}

cleanup_old_results() {
    print_step "Cleaning up old test results (keeping last 10)..."
    
    # Remove old result files, keeping only the 10 most recent
    find "$OUTPUT_DIR" -name "test_*" -type f | sort -r | tail -n +21 | xargs -r rm -f
    
    print_status "success" "Cleanup completed"
}

show_usage() {
    cat << EOF
🧠 MLX Embedding & Reranking Comprehensive Test Suite

Usage: $0 [OPTIONS]

Test Modes:
  --quick             Quick validation only (health + basic tests)
  --quality           Quality validation tests only
  --performance       Performance benchmark tests only
  --full              Full test suite (default)

Configuration:
  --url URL           Server URL (default: http://localhost:9000)
  --output-dir DIR    Output directory (default: tools/test-results)
  --help              Show this help message

Examples:
  $0                                    # Full test suite
  $0 --quick                           # Quick validation
  $0 --performance                     # Performance tests only
  $0 --url http://localhost:8080       # Custom server URL
  $0 --output-dir /tmp/test-results    # Custom output directory

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            TEST_MODE="quick"
            shift
            ;;
        --quality)
            TEST_MODE="quality"
            shift
            ;;
        --performance)
            TEST_MODE="performance"
            shift
            ;;
        --full)
            TEST_MODE="full"
            shift
            ;;
        --url)
            SERVER_URL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_header "🧠 MLX Embedding & Reranking Comprehensive Test Suite"
    print_status "info" "Test Mode: $TEST_MODE"
    print_status "info" "Server URL: $SERVER_URL"
    print_status "info" "Output Directory: $OUTPUT_DIR"
    print_status "info" "Timestamp: $TIMESTAMP"
    
    # Check prerequisites
    check_prerequisites
    
    # Check server connectivity
    if ! check_server_connectivity; then
        exit 1
    fi
    
    # Create output directory
    create_output_directory
    
    # Run tests based on mode
    local tests_passed=0
    local total_tests=0
    
    case $TEST_MODE in
        "quick")
            ((total_tests++))
            if run_quick_validation; then
                ((tests_passed++))
            fi
            ;;
        
        "quality")
            ((total_tests++))
            if run_quality_validation; then
                ((tests_passed++))
            fi
            ;;
        
        "performance")
            ((total_tests++))
            if run_performance_benchmark; then
                ((tests_passed++))
            fi
            ;;
        
        "full")
            ((total_tests += 2))
            
            if run_quality_validation; then
                ((tests_passed++))
            fi
            
            if run_performance_benchmark; then
                ((tests_passed++))
            fi
            
            # Generate comprehensive report for full tests
            generate_comprehensive_report
            ;;
    esac
    
    # Cleanup old results
    cleanup_old_results
    
    # Final summary
    print_header "🎯 Test Suite Summary"
    
    if [[ $tests_passed -eq $total_tests ]]; then
        print_status "success" "All tests passed! ($tests_passed/$total_tests)"
        print_status "success" "Your MLX embedding & reranking system is working excellently!"
    else
        print_status "warning" "Some tests failed or had issues ($tests_passed/$total_tests)"
        print_status "info" "Check the detailed logs in $OUTPUT_DIR for more information"
    fi
    
    print_status "info" "Test results saved in: $OUTPUT_DIR"
    
    # Exit with appropriate code
    if [[ $tests_passed -eq $total_tests ]]; then
        exit 0
    else
        exit 1
    fi
}

# Run main function
main "$@"
