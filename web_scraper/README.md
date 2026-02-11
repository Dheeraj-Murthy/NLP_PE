# Karnataka High Court Judgment Scraper

## Overview

This project scrapes judgments from the Karnataka High Court website (https://judiciary.karnataka.gov.in/ds_judgment.php).

## Current Status

✅ **Analysis Complete**: Website structure and flow mapped
✅ **Scraper Ready**: Full implementation with download and extraction
⚠️ **Not Implemented**: Full browser automation (click-through flow)

## What Was Discovered

### Dataset Size
- **Reportable Judgments alone**: ~300,000+ cases
- **Years Available**: 1908-2001+ (100+ years)
- **Months Per Year**: 12 months
- **Est. Total**: Millions of judgments across all categories

### Website Structure

```
ds_judgment.php
├── Tabs: Case Type, Judge, Reportable Judgments
└── Reportable Judgments
    ├── Non-Reportable (~1.3M cases)
    └── Reportable (~24K cases)
        └── Years (1908-2001+)
            ├── 1998 (~52K cases)
            ├── 1999 (~51K cases)
            └── ... (each year)
                └── Months (Jan-Dec)
                    └── Judgments (each has case #, dates, judges, parties)
```

### API Endpoints

| Endpoint | Purpose | Parameters |
|----------|----------|------------|
| `ds_type.php` | Get categories | bench, stype |
| `ds_year.php` | Get years | bench, stype, sval, htitle |
| `ds_month.php` | Get months | bench, stype, sval, yval, htitle |
| `ds_report.php` | Get judgment list | bench, stype, sval, yval, mval, htitle |
| `rep_judgment_download_single.php` | Get download URL | casenumberdata (encoded ID) |

## Files

### Main Scraper
- **`karnataka_judgment_scraper_final.py`**: Complete scraper with:
  - Full judgment metadata extraction
  - PDF download handling
  - Text extraction from PDFs
  - Structured JSON output
  - CSV indexing
  - Search HTML interface

### Documentation
- **`README_JUDGMENT_SCRAPER.md`**: Detailed analysis including:
  - Complete website structure mapping
  - All discovered API endpoints
  - Data organization strategy
  - Implementation recommendations
  - Next steps for full completion

## Usage

### Basic Scraping (API-based)

```bash
python karnataka_judgment_scraper_final.py
```

This will:
1. Scrape sample judgments (configured for testing)
2. Download PDFs
3. Extract text content
4. Save structured data to JSON/CSV
5. Create search interface

### Full Implementation (Recommended)

To scrape all ~300,000+ judgments:

**Option 1: Browser Automation (Slower but Reliable)**
```python
# Use Playwright to click through UI
# 1. Click "Reportable Judgments" tab
# 2. Click "Reportable" category
# 3. Iterate through all years
# 4. For each year, click all 12 months
# 5. For each month, navigate all judgments
# 6. Click each judgment to trigger download
# 7. Save PDF and extract text
```

**Option 2: Optimized API (Faster but Complex)**
```python
# 1. Implement proper pagination (10/25/50/100 per page)
# 2. Decode base64 download URLs
# 3. Add retry logic and rate limiting
# 4. Parallel processing with thread pools
```

## Output Structure

When scraper runs successfully, it creates:

```
karnataka_judgments_final/
├── pdfs/                    # Original PDF files
├── texts/                   # Extracted plain text
├── metadata/                 # Individual judgment JSON files
├── database/                 # Master index files
│   ├── judgments_index.csv
│   └── judgments_index.json
└── search.html              # Simple search interface
```

## Data Format

Each judgment is saved with:

```json
{
  "metadata": {
    "case_number": "WP 38262/1997",
    "date": "01-01-1998",
    "judges": "Judge Name(s)",
    "petitioner": "Petitioner Name",
    "respondent": "Respondent Name",
    "bench": "Reportable",
    "year": "1998",
    "category": "Reportable",
    "scraped_on": "2026-01-12T12:56:59Z"
  },
  "content": {
    "text": "Full judgment text...",
    "word_count": 2500,
    "has_pdf": true
  },
  "files": {
    "pdf_path": "/path/to/judgment.pdf",
    "text_path": "/path/to/judgment.txt",
    "file_size": 123456
  },
  "search_index": {
    "case_number_lower": "wp 38262/1997",
    "text_snippet": "first 500 chars...",
    "judges_lower": "judge name",
    "petitioner_lower": "petitioner",
    "respondent_lower": "respondent"
  }
}
```

## Next Steps

1. ✅ Choose scraping approach (browser vs API)
2. ✅ Implement full download handling
3. ✅ Add pagination support
4. ✅ Add rate limiting (be respectful)
5. ✅ Implement database (SQLite/PostgreSQL)
6. ✅ Create web interface for browsing
7. ✅ Add full-text search capability

## Requirements

```bash
pip install requests beautifulsoup4 pandas pdfplumber PyPDF2
```

For browser automation:
```bash
npm install playwright
npx playwright install
```

## License & Ethics

- Be respectful: Rate limit requests (1-2 seconds between downloads)
- Don't overload: Use appropriate delays
- Public data: Use responsibly, contact court for bulk access
- Consider time: Full scraping may take weeks

## Support

For issues or questions about this scraper:
- Karnataka High Court: https://judiciary.karnataka.gov.in
- Official contact: Available on website

---

**Status**: Ready for implementation | **Scrapable Cases**: 300,000+
