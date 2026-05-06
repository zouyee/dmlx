#!/bin/bash
# Generate benchmark infographic PNG from HTML
# Requires: macOS with Safari/WebKit (uses built-in tools)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HTML_FILE="$SCRIPT_DIR/dmlx-benchmark.html"
OUTPUT_FILE="$SCRIPT_DIR/dmlx-benchmark.png"

echo "Generating dmlx benchmark infographic..."
echo "  Input:  $HTML_FILE"
echo "  Output: $OUTPUT_FILE"

# Method 1: Use wkhtmltoimage if available
if command -v wkhtmltoimage &> /dev/null; then
    wkhtmltoimage --width 1200 --quality 95 "$HTML_FILE" "$OUTPUT_FILE"
    echo "✅ Generated with wkhtmltoimage: $OUTPUT_FILE"
    exit 0
fi

# Method 2: Use Chrome headless if available
if command -v "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" &> /dev/null 2>&1 || command -v chromium &> /dev/null; then
    CHROME_BIN="${CHROME_BIN:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
    if [ ! -f "$CHROME_BIN" ]; then
        CHROME_BIN="chromium"
    fi
    "$CHROME_BIN" --headless --disable-gpu --screenshot="$OUTPUT_FILE" \
        --window-size=1200,900 "file://$HTML_FILE" 2>/dev/null
    echo "✅ Generated with Chrome headless: $OUTPUT_FILE"
    exit 0
fi

# Method 3: Use Python + selenium/playwright if available
if command -v python3 &> /dev/null; then
    python3 -c "
import subprocess, sys
try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1200, 'height': 900})
        page.goto('file://$HTML_FILE')
        page.screenshot(path='$OUTPUT_FILE', full_page=True)
        browser.close()
    print('✅ Generated with Playwright: $OUTPUT_FILE')
except ImportError:
    print('⚠️  No image generation tool found.')
    print('   Install one of:')
    print('     brew install wkhtmltopdf')
    print('     pip install playwright && playwright install chromium')
    print('')
    print('   Or open the HTML file directly in a browser:')
    print(f'     open $HTML_FILE')
    sys.exit(1)
" 2>/dev/null
    exit $?
fi

echo "⚠️  No image generation tool found."
echo "   Open the HTML file in your browser and take a screenshot:"
echo "     open $HTML_FILE"
echo ""
echo "   Or install a tool:"
echo "     brew install wkhtmltopdf"
exit 1
