#!/usr/bin/env python3
"""
Karnataka High Court Judgment Scraper - Playwright-Based
Full solution using browser automation to handle downloads
"""

import asyncio
import json
import time
import re
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin
import base64

import pandas as pd
import pdfplumber
import PyPDF2

# Try to import Playwright
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("WARNING: Playwright not installed. Install with: pip install playwright && playwright install")

class KarnatakaJudgmentScraperPlaywright:
    def __init__(self):
        self.base_url = "https://judiciary.karnataka.gov.in"
        self.judgments = []
        self.output_dir = Path("karnataka_judgments_playwright")
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_dir / "pdfs").mkdir(exist_ok=True)
        (self.output_dir / "texts").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        (self.output_dir / "database").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        self.browser = None
        self.context = None
        self.page = None

        # Configuration
        self.max_judgments_per_run = 100  # For testing, can increase
        self.delay_between_clicks = 0.5
        self.delay_between_downloads = 2
        self.headless = False  # Set to True for production

    def __enter__(self):
        self.start_browser()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_browser()
        return False

    def start_browser(self):
        print("Starting browser...")

        self.playwright = sync_playwright().start()

        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            slow_mo=500
        )

        self.context = self.browser.new_context(
            accept_downloads=True,
            viewport={'width': 1920, 'height': 1080}
        )

        self.page = self.context.new_page()

        # Downloads
        self.downloads_dir = self.output_dir / "pdfs" / "temp"
        self.downloads_dir.mkdir(parents=True, exist_ok=True)

        print("Browser started successfully")

    def close_browser(self):
        try:
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if hasattr(self, "playwright"):
                self.playwright.stop()
        except:
            pass
        print("Browser closed")

    def _handle_response(self, response):
        """Handle HTTP responses to track downloads"""
        # Could be used to track download URLs
        pass

    def navigate_to_judgments(self):
        """Navigate to judgments page"""
        print(f"Navigating to {self.base_url}/ds_judgment.php...")
        self.page.goto(f"{self.base_url}/ds_judgment.php", wait_until="networkidle", timeout=60000)
        print("Navigation complete")
        time.sleep(2)

    def click_element_by_text(self, text, timeout=10000):
        """Click on element containing specific text"""
        try:
            print(f"Looking for element with text: {text[:50]}...")
            # Try multiple selector strategies
            selectors = [
                f"text={text}",
                f"button:has-text(\"{text}\")",
                f"a:has-text(\"{text}\")",
            ]

            for selector in selectors:
                try:
                    element = self.page.locator(selector).first
                    if element.count() > 0:
                        element.click(timeout=timeout)
                        time.sleep(self.delay_between_clicks)
                        print(f"Clicked on: {text}")
                        return True
                except:
                    continue

            print(f"Could not find element with text: {text}")
            return False

        except Exception as e:
            print(f"Error clicking on element with text {text}: {e}")
            return False

    def wait_for_element(self, selector, timeout=10000):
        """Wait for element to appear"""
        try:
            element = self.page.locator(selector).first
            element.wait_for(state="visible", timeout=timeout)
            return element
        except Exception as e:
            print(f"Error waiting for element {selector}: {e}")
            return None

    def get_current_url(self):
        """Get current page URL"""
        return self.page.url

    def take_screenshot(self, name):
        """Take screenshot for debugging"""
        screenshot_path = self.output_dir / "logs" / f"{name}.png"
        self.page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"Screenshot saved: {screenshot_path}")

    def scrape_reportable_judgments(self, max_years=2, max_months_per_year=2, max_judgments_per_month=3):
        """Scrape reportable judgments"""
        print("\n" + "="*80)
        print("STARTING REPORTABLE JUDGMENT SCRAPING")
        print("="*80)

        try:
            # Step 1: Navigate to page
            self.navigate_to_judgments()
            self.take_screenshot("01_initial_page")

            # Step 2: Click on "Reportable Judgments" tab
            print("\n[Step 2/8] Clicking 'Reportable Judgments' tab...")
            if not self.click_element_by_text("Reportable Judgments"):
                print("Failed to click Reportable Judgments tab")
                return False

            time.sleep(1)
            self.take_screenshot("02_after_tab_click")

            # Step 3: Click on "Reportable" category
            print("\n[Step 3/8] Clicking 'Reportable' category...")
            if not self.click_element_by_text("Reportable"):
                print("Failed to click Reportable category")
                return False

            time.sleep(1)
            self.take_screenshot("03_after_category_click")

            # Step 4: Get available years
            print("\n[Step 4/8] Getting available years...")
            self.take_screenshot("04_years_list")

            # Look for year buttons and collect them
            years = []
            year_buttons = self.page.locator("button").all()

            for i, button in enumerate(year_buttons):
                try:
                    text = button.text_content()
                    onclick = button.get_attribute('onclick')

                    if onclick and 'secondfunction' in onclick:
                        match = re.search(r"secondfunction\('3','([^']+)','([^']+)'\)", onclick)
                        if match:
                            year_code = match.group(2).strip()
                            year_name = text.split('[')[0].strip()

                            # Extract count from text
                            count_match = re.search(r'\[(\d+)\]', text)
                            count = int(count_match.group(1)) if count_match else 0

                            years.append({
                                'code': year_code,
                                'name': year_name,
                                'count': count,
                                'button_element': button,
                                'index': i
                            })
                            print(f"  Found year {len(years)}: {year_name} ({count} judgments)")
                except Exception as e:
                    print(f"Error parsing year button: {e}")
                    continue

            if not years:
                print("No years found")
                return False

            # Sort by count (descending) and limit
            years.sort(key=lambda x: x['count'], reverse=True)
            years = years[:max_years]

            print(f"\nProcessing {len(years)} years (top {max_years} by count)")

            # Process each year
            for year_idx, year_data in enumerate(years):
                print(f"\n" + "-"*80)
                print(f"YEAR {year_idx+1}/{len(years)}: {year_data['name']} ({year_data['count']:,} judgments)")
                print("-"*80)

                # Step 5: Click on year
                print(f"[Step 5/8] Clicking on year: {year_data['name']}...")

                try:
                    year_data['button_element'].click(timeout=10000)
                    time.sleep(self.delay_between_clicks * 2)
                    print(f"  Clicked on year")
                except Exception as e:
                    print(f"  Error clicking on year: {e}")
                    continue

                self.take_screenshot(f"05_year_{year_data['code']}")

                # Step 6: Get available months
                print(f"[Step 6/8] Getting months for {year_data['name']}...")
                time.sleep(1)

                months = []
                month_buttons = self.page.locator("button").all()

                for button in month_buttons:
                    try:
                        text = button.text_content()
                        onclick = button.get_attribute('onclick')

                        if onclick and 'thirdfunction' in onclick:
                            match = re.search(r"thirdfunction\('3','([^']+)','([^']+)','([^']+)'\)", onclick)
                            if match:
                                month_code = match.group(3).strip()
                                month_name = text.split('-')[0].strip()
                                count_match = re.search(r'\[(\d+)\]', text)
                                count = int(count_match.group(1)) if count_match else 0

                                months.append({
                                    'code': month_code,
                                    'name': month_name,
                                    'count': count,
                                    'button_element': button
                                })
                    except Exception as e:
                        continue

                if not months:
                    print(f"  No months found for {year_data['name']}")
                    continue

                # Sort by count and limit
                months.sort(key=lambda x: x['count'], reverse=True)
                months = months[:max_months_per_year]

                print(f"  Found {len(months)} months")

                # Process each month
                for month_idx, month_data in enumerate(months):
                    print(f"\n    MONTH {month_idx+1}/{len(months)}: {month_data['name']} ({month_data['count']:,} judgments)")
                    print(f"    [Step 7/8] Clicking on month: {month_data['name']}...")

                    try:
                        month_data['button_element'].click(timeout=10000)

                        # 🔥 CRITICAL: wait for fourthfunction() to finish loading cases
                        self.page.wait_for_selector("table tbody tr", timeout=20000)

                        print("      ✓ Case table loaded")

                    except Exception as e:
                        print(f"      Error clicking month or waiting for table: {e}")
                        continue

                    # Step 8: Read the loaded case table
                    print(f"    [Step 8/8] Reading judgment list...")

                    judgments_collected = self._collect_judgments_from_table(
                        year_data['name'],
                        month_data['name'],
                        max_judgments=max_judgments_per_month
                    )

                    if judgments_collected:
                        print(f"      Found {len(judgments_collected)} judgments")

                        # Step 9: Click on each judgment to trigger download
                        for j, judgment_data in enumerate(judgments_collected):  # Test with 3
                            print(f"      Processing judgment {j+1}/{len(judgments_collected[:3])}: {judgment_data['case_number']}")

                            # Click on judgment
                            try:
                                judgment_data['button_element'].click(timeout=10000)
                                time.sleep(2)  # Wait for download

                                # Try to find and handle download
                                downloaded = self._wait_for_and_handle_download(judgment_data)

                                if downloaded:
                                    print(f"        ✓ Downloaded successfully")
                                else:
                                    print(f"        ✗ Download may have failed")

                                # Go back to judgment list
                                time.sleep(0.5)
                                self.page.go_back()
                                time.sleep(0.5)

                            except Exception as e:
                                print(f"      Error with judgment {judgment_data['case_number']}: {e}")
                                continue

                    # Go back to year list
                    print(f"    Going back to year list...")
                    self.page.go_back()
                    time.sleep(1)

                # Go back to year list after processing months
                print(f"Going back to year list...")
                self.page.go_back()
                time.sleep(1)

            return True

        except Exception as e:
            print(f"Error during scraping: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _collect_judgments_from_table(self, year_name, month_name, max_judgments=20):
        """Collect judgment data from table"""
        judgments = []

        try:
            # Wait for table to load
            time.sleep(1)

            # Look for the main judgment table
            # Try different table selectors
            tables = self.page.locator("table").all()

            main_table = None
            for table in tables:
                row_count = len(table.locator("tbody tr").all())
                if row_count > 10:  # Likely the main judgment table
                    main_table = table
                    break

            if not main_table:
                print("      No judgment table found")
                return judgments

            # Get all rows except header
            rows = main_table.locator("tbody tr").all()

            # Limit to max_judgments + 1 (for header)
            rows_to_process = rows[1:min(max_judgments + 1, len(rows))]

            print(f"      Processing {len(rows_to_process)} rows...")

            for i, row in enumerate(rows_to_process):
                try:
                    cells = row.locator("td, th").all()

                    if len(cells) >= 4:
                        date_cell = cells[0]
                        title_cell = cells[1]
                        judges_cell = cells[2]
                        petitioner_cell = cells[3]
                        respondent_cell = cells[4]

                        # Get text content
                        date_text = date_cell.text_content(timeout=5000)
                        judges_text = judges_cell.text_content(timeout=5000)
                        petitioner_text = petitioner_cell.text_content(timeout=5000)
                        respondent_text = respondent_cell.text_content(timeout=5000)

                        # Get case number from button in title cell
                        button = title_cell.locator("button").first()
                        case_number_text = button.text_content(timeout=5000)
                        case_id = button.get_attribute('id')

                        if i < 3:  # Only log first 3
                            print(f"        {i+1}. {case_number_text}")

                        judgment_data = {
                            'case_id': case_id,
                            'case_number': case_number_text.strip(),
                            'date': date_text.strip(),
                            'judges': judges_text.strip(),
                            'petitioner': petitioner_text.strip(),
                            'respondent': respondent_text.strip(),
                            'year': year_name,
                            'month': month_name,
                            'button_element': button
                        }

                        judgments.append(judgment_data)

                except Exception as e:
                    print(f"      Error processing row {i}: {e}")
                    continue

            return judgments

        except Exception as e:
            print(f"Error collecting judgments: {e}")
            return []

    def _wait_for_and_handle_download(self, judgment_data):
        """Wait for download and handle it"""
        try:
            # Wait a bit for download to start
            print(f"        Waiting for download...")

            # Method 1: Check for new file in downloads directory
            downloads_dir = self.output_dir / "pdfs" / "temp"

            initial_files = list(downloads_dir.glob("*")) if downloads_dir.exists() else []

            # Wait up to 10 seconds for download
            max_wait = 10
            for i in range(max_wait):
                time.sleep(1)
                current_files = list(downloads_dir.glob("*")) if downloads_dir.exists() else []
                new_files = [f for f in current_files if f not in initial_files]

                if new_files:
                    # Found a new file
                    new_file = new_files[0]
                    print(f"        Download started: {new_file.name}")

                    # Wait for download to complete (file size stops changing)
                    stable_count = 0
                    last_size = 0

                    for wait_iter in range(15):
                        time.sleep(0.5)

                        if new_file.exists():
                            try:
                                current_size = new_file.stat().st_size
                                if current_size > 0 and abs(current_size - last_size) < 100:  # Stable for 0.5s
                                    stable_count += 1
                                if stable_count >= 3:
                                    break
                                last_size = current_size
                            except:
                                pass

                    if new_file.exists():
                        # Move to final location
                        final_path = self.output_dir / "pdfs" / f"{self._safe_filename(judgment_data['case_number'])}.pdf"
                        new_file.rename(final_path)

                        # Extract text
                        text = self._extract_text_from_pdf(final_path)

                        # Save metadata
                        self._save_judgment_metadata(judgment_data, final_path, text)

                        print(f"        ✓ Saved: {final_path.name} ({len(text)} chars)")

                        # Clean up temp
                        for temp_file in downloads_dir.glob("*"):
                            try:
                                temp_file.unlink()
                            except:
                                pass

                        return True

            print(f"        No download detected after {max_wait} seconds")
            return False

        except Exception as e:
            print(f"        Error handling download: {e}")
            return False

    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            return text

        except Exception as e:
            # Try PyPDF2 as fallback
            try:
                text = ""
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"

                return text
            except Exception as e2:
                print(f"        Error extracting text: {e2}")
                return f"Extraction failed: {e}"

    def _save_judgment_metadata(self, judgment_data, pdf_path, text):
        """Save judgment metadata to JSON and update index"""

        safe_case_number = self._safe_filename(judgment_data['case_number'])

        # Create structured metadata
        metadata = {
            'case_number': judgment_data['case_number'],
            'date': judgment_data['date'],
            'judges': judgment_data['judges'],
            'petitioner': judgment_data['petitioner'],
            'respondent': judgment_data['respondent'],
            'year': judgment_data['year'],
            'month': judgment_data['month'],
            'bench': 'Reportable',
            'scraped_on': datetime.now().isoformat(),
            'source': 'Karnataka High Court',
            'pdf_path': str(pdf_path),
            'file_size': pdf_path.stat().st_size if pdf_path.exists() else 0,
            'text_length': len(text) if text else 0,
            'text_preview': text[:500] if text else ""
        }

        # Save individual JSON
        json_path = self.output_dir / "metadata" / f"{safe_case_number}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Save text
        if text:
            txt_path = self.output_dir / "texts" / f"{safe_case_number}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)

        # Add to master index
        self.judgments.append(metadata)

        # Save master index every 10 judgments
        if len(self.judgments) % 10 == 0:
            self._save_master_index()

    def _safe_filename(self, text):
        """Create safe filename from text"""
        # Remove invalid characters
        safe = re.sub(r'[\\/*?:"<>|]', '', text)
        safe = re.sub(r'\s+', '_', safe)
        return safe[:100]  # Limit length

    def _save_master_index(self):
        """Save master index files"""
        if not self.judgments:
            return

        # Save CSV
        df = pd.DataFrame(self.judgments)
        csv_path = self.output_dir / "database" / "judgments_index.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')

        # Save JSON
        json_path = self.output_dir / "database" / "judgments_index.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.judgments, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"MASTER INDEX UPDATED: {len(self.judgments)} judgments total")
        print(f"CSV: {csv_path}")
        print(f"JSON: {json_path}")
        print(f"{'='*80}")

    def create_search_html(self):
        """Create HTML search interface"""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Karnataka High Court Judgments - Search</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 20px auto;
            line-height: 1.6;
        }}
        h1 {{
            color: #183a52;
            border-bottom: 3px solid #183a52;
            padding-bottom: 20px;
        }}
        .stats {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .search-box {{
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-sizing: border-box;
        }}
        .judgment {{
            border: 1px solid #ddd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            transition: background-color 0.3s;
        }}
        .judgment:hover {{
            background-color: #f0f0f0;
        }}
        .case-number {{
            font-weight: bold;
            color: #183a52;
            font-size: 18px;
        }}
        .date {{
            color: #666;
            font-size: 14px;
        }}
        .text-preview {{
            color: #333;
            font-size: 14px;
            max-height: 100px;
            overflow-y: auto;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 12px;
            color: white;
            margin-left: 10px;
        }}
        .badge-pdf {{
            background-color: #28a745;
        }}
        .badge-text {{
            background-color: #6c757d;
        }}
        a {{
            color: #183a52;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <h1>Karnataka High Court Judgments Search</h1>

    <div class="stats">
        <h3>Statistics</h3>
        <p><strong>Total Judgments:</strong> {len(self.judgments)}</p>
        <p><strong>Bench:</strong> Reportable</p>
        <p><strong>Last Updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>

    <input type="text" id="search" class="search-box" placeholder="Search by case number, judge, petitioner, respondent, or text content...">

    <div id="results"></div>

    <script>
        const judgments = {json.dumps(self.judgments, indent=2)};

        document.getElementById('search').addEventListener('input', function(e) {{
            const searchTerm = e.target.value.toLowerCase();
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '';

            if (searchTerm.length < 2) {{
                resultsDiv.innerHTML = '<p style="text-align:center;color:#666;">Type at least 2 characters to search...</p>';
                return;
            }}

            const filtered = judgments.filter(j => 
                j.case_number.toLowerCase().includes(searchTerm) ||
                j.judges.toLowerCase().includes(searchTerm) ||
                j.petitioner.toLowerCase().includes(searchTerm) ||
                j.respondent.toLowerCase().includes(searchTerm) ||
                (j.text_preview && j.text_preview.toLowerCase().includes(searchTerm))
            );

            if (filtered.length === 0) {{
                resultsDiv.innerHTML = '<p style="text-align:center;color:#666;">No judgments found matching your search.</p>';
                return;
            }}

            filtered.forEach(j => {{
                const div = document.createElement('div');
                div.className = 'judgment';
                div.innerHTML = `
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span class="case-number">${{j.case_number}}</span>
                        <span class="date">${{j.date}}</span>
                    </div>
                    <div><strong>Judges:</strong> ${{j.judges}}</div>
                    <div><strong>Petitioner:</strong> ${{j.petitioner}}</div>
                    <div><strong>Respondent:</strong> ${{j.respondent}}</div>
                    ${{j.text_preview ? '<div class="text-preview">' + j.text_preview.substring(0, 300) + '...</div>' : ''}}
                    <div>
                        <a href="${{j.pdf_path}}" target="_blank" style="font-size:12px;">Open PDF</a>
                        <span class="badge badge-pdf">PDF</span>
                        ${{j.text_path ? '<a href="${{j.text_path}}" target="_blank">View Text</a>' : '<span class="badge badge-text">No Text</span>'}}
                    </div>
                `;
                resultsDiv.appendChild(div);
            }});
        }});
    </script>
</body>
</html>
"""

        html_path = self.output_dir / "search.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"Search interface created: {html_path}")

    def run(self, max_years=2, max_months_per_year=2, max_judgments_per_month=3):
        """Run the scraper"""
        print(f"\n{'='*80}")
        print(f"KARNATAKA HIGH COURT JUDGMENT SCRAPER - PLAYWRIGHT EDITION")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  - Max years: {max_years}")
        print(f"  - Max months per year: {max_months_per_year}")
        print(f"  - Max judgments per month: {max_judgments_per_month}")
        print(f"  - Headless mode: {self.headless}")
        print(f"  - Output directory: {self.output_dir}")
        print(f"{'='*80}")

        start_time = time.time()

        try:
            self.start_browser()
            success = self.scrape_reportable_judgments(
                max_years=max_years,
                max_months_per_year=max_months_per_year,
                max_judgments_per_month=max_judgments_per_month
            )

            if success:
                # Create search interface
                self.create_search_html()

                elapsed = time.time() - start_time

                print(f"\n{'='*80}")
                print(f"SCRAPING COMPLETE")
                print(f"{'='*80}")
                print(f"Total Time: {elapsed:.1f} seconds")
                print(f"Judgments Scraped: {len(self.judgments)}")
                print(f"{'='*80}")
                print(f"\nOutput:")
                print(f"  PDFs: {self.output_dir / 'pdfs'}")
                print(f"  Texts: {self.output_dir / 'texts'}")
                print(f"  Metadata: {self.output_dir / 'metadata'}")
                print(f"  Database: {self.output_dir / 'database'}")
                print(f"  Search: {self.output_dir / 'search.html'}")
                print(f"{'='*80}")
                print("\nNext Steps:")
                print("1. Review downloaded judgments in output directory")
                print("2. Use search.html to browse and search judgments")
                print("3. Increase max_years/max_months for full coverage")
                print("4. Consider running with headless=True for production")
            else:
                print("\nScraping failed. Check logs for errors.")

        except Exception as e:
            print(f"\nFATAL ERROR: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.close_browser()

def main():
    """Main entry point"""
    print("Karnataka High Court Judgment Scraper")
    print("=" * 80)
    print("Press Enter to start scraping...")
    print("=" * 80)
    input()

    # Run scraper with conservative limits for testing
    scraper = KarnatakaJudgmentScraperPlaywright()
    scraper.run(
        max_years=2,        # First 2 years (for testing)
        max_months_per_year=2,  # First 2 months per year
        max_judgments_per_month=3   # First 3 judgments per month
    )

if __name__ == "__main__":
    main()
