#!/bin/bash

# RAG Pipeline Test Runner Script
# This script simplifies running the comprehensive RAG pipeline tests

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODE="full"
DEBUG=""
REPORT_FILE=""
CHECK_SERVICES=true

# Help function
show_help() {
    cat << EOF
RAG Pipeline Test Runner

Usage: $0 [OPTIONS]

OPTIONS:
    -m, --mode MODE         Test mode: quick, full, or stress (default: full)
    -d, --debug            Enable debug logging
    -r, --report FILE      Save test report to JSON file
    -s, --skip-services    Skip service health checks
    -h, --help             Show this help message

EXAMPLES:
    $0                              # Run full test suite
    $0 -m quick                     # Run quick tests only
    $0 -m stress -r results.json    # Run stress tests with report
    $0 -d -r debug_report.json      # Run with debug logging

PREREQUISITES:
    - Docker containers running (TEI, Qdrant)
    - Virtual environment activated
    - Environment variables configured

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -d|--debug)
            DEBUG="--debug"
            shift
            ;;
        -r|--report)
            REPORT_FILE="--report-file $2"
            shift 2
            ;;
        -s|--skip-services)
            CHECK_SERVICES=false
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Validate mode
case $MODE in
    quick|full|stress)
        ;;
    *)
        echo -e "${RED}Invalid mode: $MODE. Must be quick, full, or stress${NC}"
        exit 1
        ;;
esac

echo -e "${BLUE}🕷️  RAG Pipeline Test Runner${NC}"
echo -e "${BLUE}=============================${NC}"
echo ""

# Check if we're in the right directory
if [[ ! -f "tests/test_full_rag_pipeline.py" ]]; then
    echo -e "${RED}Error: Must be run from the crawl-mcp project root directory${NC}"
    echo -e "${YELLOW}Current directory: $(pwd)${NC}"
    exit 1
fi

# Check virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}Warning: No virtual environment detected${NC}"
    echo -e "${YELLOW}Consider running: uv venv && source .venv/bin/activate${NC}"
    echo ""
fi

# Check environment file
if [[ ! -f ".env" ]]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo -e "${YELLOW}Consider copying .env.example to .env and configuring it${NC}"
    echo ""
fi

# Service health checks
if [[ "$CHECK_SERVICES" == true ]]; then
    echo -e "${BLUE}Checking service prerequisites...${NC}"

    # Check if TEI service is running
    if command -v curl &> /dev/null; then
        TEI_URL="${TEI_URL:-http://localhost:8080}"
        echo -n "  TEI Service ($TEI_URL): "
        if curl -s "$TEI_URL/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Running${NC}"
        else
            echo -e "${YELLOW}⚠ Not responding${NC}"
            echo -e "${YELLOW}    Consider running: docker-compose up tei${NC}"
        fi

        # Check if Qdrant service is running
        QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
        echo -n "  Qdrant Service ($QDRANT_URL): "
        if curl -s "$QDRANT_URL/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Running${NC}"
        else
            echo -e "${YELLOW}⚠ Not responding${NC}"
            echo -e "${YELLOW}    Consider running: docker-compose up qdrant${NC}"
        fi
    else
        echo -e "${YELLOW}  curl not available, skipping service checks${NC}"
    fi
    echo ""
fi

# Display test configuration
echo -e "${BLUE}Test Configuration:${NC}"
echo -e "  Mode: ${YELLOW}$MODE${NC}"
echo -e "  Debug: ${YELLOW}${DEBUG:-disabled}${NC}"
echo -e "  Report: ${YELLOW}${REPORT_FILE:-none}${NC}"
echo ""

# Estimate runtime
case $MODE in
    quick)
        echo -e "${BLUE}Estimated runtime: ${YELLOW}~2 minutes${NC}"
        ;;
    full)
        echo -e "${BLUE}Estimated runtime: ${YELLOW}~10 minutes${NC}"
        ;;
    stress)
        echo -e "${BLUE}Estimated runtime: ${YELLOW}~20+ minutes${NC}"
        ;;
esac
echo ""

# Prompt for confirmation in stress mode
if [[ "$MODE" == "stress" ]]; then
    read -p "Stress testing may generate significant load. Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Test cancelled by user${NC}"
        exit 0
    fi
fi

# Run the tests
echo -e "${GREEN}Starting RAG pipeline tests...${NC}"
echo ""

# Build the command
CMD="python tests/test_full_rag_pipeline.py --mode $MODE $DEBUG $REPORT_FILE"

# Execute the tests
if eval $CMD; then
    echo ""
    echo -e "${GREEN}🎉 Tests completed successfully!${NC}"

    # Show report location if generated
    if [[ -n "$REPORT_FILE" ]]; then
        REPORT_PATH=$(echo $REPORT_FILE | cut -d' ' -f2)
        echo -e "${BLUE}📊 Test report saved to: ${YELLOW}$REPORT_PATH${NC}"
    fi

    exit 0
else
    echo ""
    echo -e "${RED}❌ Tests failed. Check the output above for details.${NC}"

    # Suggest common fixes
    echo ""
    echo -e "${YELLOW}Common fixes:${NC}"
    echo -e "  • Ensure Docker services are running: ${BLUE}docker-compose up -d${NC}"
    echo -e "  • Check environment variables in .env file"
    echo -e "  • Verify network connectivity to external test URLs"
    echo -e "  • Run with --debug for more detailed error information"

    exit 1
fi
