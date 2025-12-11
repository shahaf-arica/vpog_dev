#!/bin/bash
# Quick test script for VPOG dataloader components
# Run this to verify all unit tests pass

echo "=========================================="
echo "VPOG Dataloader Quick Test"
echo "=========================================="
echo ""

# Set project root
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"

echo "Project root: $PROJECT_ROOT"
echo ""

# Test 1: Template Selector
echo "=========================================="
echo "Test 1: Template Selector"
echo "=========================================="
python training/dataloader/template_selector.py
if [ $? -ne 0 ]; then
    echo "❌ Template selector test FAILED"
    exit 1
fi
echo ""

# Test 2: Flow Computer
echo "=========================================="
echo "Test 2: Flow Computer"
echo "=========================================="
python training/dataloader/flow_computer.py
if [ $? -ne 0 ]; then
    echo "❌ Flow computer test FAILED"
    exit 1
fi
echo ""

# Test 3: Visualization Utils
echo "=========================================="
echo "Test 3: Visualization Utils"
echo "=========================================="
python training/dataloader/vis_utils.py
if [ $? -ne 0 ]; then
    echo "❌ Visualization test FAILED"
    exit 1
fi
echo ""

# Test 4: Integration Test (optional - requires data)
echo "=========================================="
echo "Test 4: Integration Test"
echo "=========================================="
echo "This test requires GSO data. Attempting..."
python training/dataloader/test_integration.py
if [ $? -ne 0 ]; then
    echo "⚠️  Integration test FAILED (may be due to missing data)"
    echo "    This is OK if you don't have GSO data yet."
else
    echo "✓ Integration test PASSED"
fi
echo ""

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "✓ Template Selector: PASSED"
echo "✓ Flow Computer: PASSED"
echo "✓ Visualization: PASSED"
echo ""
echo "Check tmp/ directory for test outputs"
echo ""
echo "Next steps:"
echo "1. Review visualizations in tmp/"
echo "2. If you have data, check integration test results"
echo "3. Read VPOG_SUMMARY.md for next steps"
echo "=========================================="
