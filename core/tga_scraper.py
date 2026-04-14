"""
TGA Qualification Scraper
=========================
Reads 1150+ qualification codes from the source Excel file, then for each:
  1. Scrapes training.gov.au/Training/Details/{code}/summary for:
       - ANZSCO Identifier
       - Taxonomy – Industry Sector
       - Taxonomy – Occupation
  2. Scrapes training.gov.au/Training/Details/{code} (units tab) for:
       - Core units  (code + title)
       - Elective units (code + title, with group label where applicable)

Outputs a new Excel file with one row per qualification containing all the
above, plus the original columns from the source file.

RESUME SUPPORT
--------------
Progress is saved to  tga_scraper_checkpoint.csv  after every qualification.
If the script is interrupted, re-running it will skip already-scraped codes.

RATE LIMITING
-------------
A 2-second sleep is enforced between requests. TGA will block your IP if you
go faster. For 1150 qualifications this takes ~40 minutes.

USAGE
-----
    pip install requests beautifulsoup4 pandas openpyxl tqdm
    python tga_scraper.py

    # To use a different source file:
    python tga_scraper.py --input my_file.xlsx --output results.xlsx
"""

import argparse
import os
import re
import sys
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── Optional progress bar ────────────────────────────────────────────────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Constants ────────────────────────────────────────────────────────────────
BASE_URL      = "https://training.gov.au/Training/Details"
SLEEP_SECONDS = 2          # polite crawl delay – do not lower below 2
TIMEOUT       = 20
CHECKPOINT    = "tga_scraper_checkpoint.csv"

def _normalise_taxonomy_value(raw: str) -> str:
    """
    Normalise a taxonomy cell value so multiple entries are always
    comma-separated, regardless of how TGA originally delimited them
    (newlines, semicolons, bullet characters, etc.).

    E.g.  "Animal Attendants\nVeterinary Nurses"
          → "Animal Attendants, Veterinary Nurses"
    """
    if not raw or raw == "N/A":
        return raw
    # Split on common delimiters: newline, semicolon, pipe, bullet
    parts = re.split(r"[\n\r;|•]+", raw)
    # Strip whitespace and drop empty strings
    parts = [p.strip() for p in parts if p.strip()]
    return ", ".join(parts)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-AU,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _get(url: str) -> BeautifulSoup | None:
    """Fetch a URL and return a BeautifulSoup object, or None on failure."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return BeautifulSoup(r.content, "html.parser")
    except Exception as exc:
        print(f"    ⚠  GET failed: {url}  →  {exc}", file=sys.stderr)
        return None


# ── Taxonomy scraper ─────────────────────────────────────────────────────────

def scrape_taxonomy(qual_code: str) -> dict:
    """
    Returns a dict with keys:
        ANZSCO_Identifier, Taxonomy_Industry_Sector, Taxonomy_Occupation
    Values default to 'N/A' on failure.
    """
    url = f"{BASE_URL}/{qual_code}/summary"
    result = {
        "ANZSCO_Identifier":       "N/A",
        "Taxonomy_Industry_Sector": "N/A",
        "Taxonomy_Occupation":      "N/A",
    }

    soup = _get(url)
    if soup is None:
        return result

    # ── Strategy 1: Classifications table ────────────────────────────────────
    class_header = soup.find(
        ["h2", "h3"],
        string=lambda t: t and "classifications" in t.lower()
    )
    if class_header:
        table = class_header.find_next("table")
        if table:
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2:
                    # Some pages use 2 cols (label | value), others 3 cols
                    scheme = cells[0].get_text(strip=True)
                    value  = cells[-1].get_text(strip=True)   # last cell = value

                    if "ANZSCO" in scheme:
                        result["ANZSCO_Identifier"] = value
                    elif "Industry Sector" in scheme:
                        result["Taxonomy_Industry_Sector"] = value
                    elif "Occupation" in scheme:
                        result["Taxonomy_Occupation"] = value

    # ── Strategy 2: definition-list fallback ─────────────────────────────────
    if all(v == "N/A" for v in result.values()):
        for dt in soup.find_all("dt"):
            label = dt.get_text(strip=True)
            dd = dt.find_next_sibling("dd")
            if dd:
                val = dd.get_text(strip=True)
                if "ANZSCO" in label:
                    result["ANZSCO_Identifier"] = val
                elif "Industry Sector" in label:
                    result["Taxonomy_Industry_Sector"] = val
                elif "Occupation" in label:
                    result["Taxonomy_Occupation"] = val

    # Normalise all three fields to consistent comma-separated strings
    for key in result:
        result[key] = _normalise_taxonomy_value(result[key])

    return result


# ── Units scraper ─────────────────────────────────────────────────────────────

# Regex to identify a unit code at the start of a text node
_UNIT_CODE_RE = re.compile(
    r"^([A-Z]{2,8}\d{3,6}[A-Z]?)\s+(.*)",   # e.g.  ACMGEN101  Explore job...
    re.DOTALL
)

def _extract_units_from_table(table) -> list[dict]:
    """
    Parse a <table> whose rows contain unit codes + titles.
    Returns list of  {'unit_code': ..., 'unit_title': ..., 'group': ...}
    """
    units  = []
    group  = ""
    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        text = " ".join(c.get_text(" ", strip=True) for c in cells)

        # Group / section headers (e.g. "Group A – Aged care")
        if len(cells) == 1 and not _UNIT_CODE_RE.match(text):
            group = text
            continue

        # Try first cell as code, second as title
        code  = cells[0].get_text(strip=True)
        title = cells[1].get_text(strip=True) if len(cells) > 1 else ""

        if re.match(r"^[A-Z]{2,8}\d{3,6}[A-Z]?$", code):
            units.append({"unit_code": code, "unit_title": title, "group": group})
            continue

        # Fallback: combined text in one cell
        m = _UNIT_CODE_RE.match(text.strip())
        if m:
            units.append({
                "unit_code":  m.group(1),
                "unit_title": m.group(2).strip(),
                "group":      group,
            })

    return units


def _extract_units_from_list(tag) -> list[dict]:
    """
    Parse <ul>/<ol> blocks that contain unit codes.
    """
    units = []
    for li in tag.find_all("li"):
        text = li.get_text(" ", strip=True)
        m = _UNIT_CODE_RE.match(text)
        if m:
            units.append({
                "unit_code":  m.group(1),
                "unit_title": m.group(2).strip(),
                "group":      "",
            })
    return units


def scrape_units(qual_code: str) -> dict:
    """
    Scrapes core and elective units from the TGA qualification page.

    Returns:
        {
            'core_units':     [ {'unit_code': ..., 'unit_title': ...}, ... ],
            'elective_units': [ {'unit_code': ..., 'unit_title': ..., 'group': ...}, ... ],
        }
    """
    result = {"core_units": [], "elective_units": []}

    # Try the dedicated summary/packaging-rules page first
    for path_suffix in ["", "/summary"]:
        url  = f"{BASE_URL}/{qual_code}{path_suffix}"
        soup = _get(url)
        if soup is None:
            continue

        # ── Find Core and Elective headings ───────────────────────────────────
        section_map = {}   # 'core' / 'elective' → list of units

        for heading in soup.find_all(["h2", "h3", "h4", "strong", "b"]):
            text_low = heading.get_text(strip=True).lower()

            if "core unit" in text_low:
                section_key = "core"
            elif "elective unit" in text_low or "group" in text_low:
                section_key = "elective"
            else:
                continue

            # Determine the group label for elective groups
            group_label = heading.get_text(strip=True) if "group" in text_low else ""

            # Look for the content that follows this heading
            sibling = heading.find_next_sibling()
            while sibling:
                tag_name = sibling.name
                if tag_name in ["h2", "h3", "h4"] and sibling != heading:
                    # Stop at next major heading
                    break

                units_found = []
                if tag_name == "table":
                    units_found = _extract_units_from_table(sibling)
                elif tag_name in ["ul", "ol"]:
                    units_found = _extract_units_from_list(sibling)
                elif tag_name in ["div", "section"]:
                    # Try nested table or list
                    inner = sibling.find("table")
                    if inner:
                        units_found = _extract_units_from_table(inner)
                    else:
                        inner = sibling.find(["ul", "ol"])
                        if inner:
                            units_found = _extract_units_from_list(inner)
                        else:
                            # Plain text parsing
                            raw = sibling.get_text("\n", strip=True)
                            for line in raw.splitlines():
                                m = _UNIT_CODE_RE.match(line.strip())
                                if m:
                                    units_found.append({
                                        "unit_code":  m.group(1),
                                        "unit_title": m.group(2).strip(),
                                        "group":      group_label,
                                    })

                # Tag the group label on elective units
                for u in units_found:
                    if section_key == "elective" and not u.get("group"):
                        u["group"] = group_label

                if units_found:
                    section_map.setdefault(section_key, []).extend(units_found)

                sibling = sibling.find_next_sibling()

        if section_map:
            result["core_units"]     = section_map.get("core", [])
            result["elective_units"] = section_map.get("elective", [])
            break   # Success – don't need the second URL

        time.sleep(SLEEP_SECONDS)

    return result


# ── Formatting helpers ────────────────────────────────────────────────────────

def units_to_string(units: list[dict], include_group: bool = False) -> str:
    """
    Convert a list of unit dicts to a human-readable multi-line string.
    E.g.:
        ACMGEN101 | Explore job opportunities in animal care
        ACMGEN102 | Approach and handle a range of calm animals
    """
    lines = []
    current_group = None
    for u in units:
        g = u.get("group", "")
        if include_group and g and g != current_group:
            lines.append(f"[{g}]")
            current_group = g
        lines.append(f"{u['unit_code']} | {u['unit_title']}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(input_path: str, output_path: str) -> None:

    # ── Load source data ──────────────────────────────────────────────────────
    print(f"Loading: {input_path}")
    df = pd.read_excel(input_path)
    qual_codes = df["Qualification Code"].dropna().unique().tolist()
    print(f"Found {len(qual_codes)} unique qualification codes.\n")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    done: dict[str, dict] = {}
    if os.path.exists(CHECKPOINT):
        chk = pd.read_csv(CHECKPOINT)
        for _, row in chk.iterrows():
            done[row["Qualification_Code"]] = row.to_dict()
        print(f"Checkpoint loaded — {len(done)} already scraped, "
              f"{len(qual_codes) - len(done)} remaining.\n")

    # ── Scrape ────────────────────────────────────────────────────────────────
    remaining = [c for c in qual_codes if c not in done]
    iterator  = tqdm(remaining, unit="qual") if HAS_TQDM else remaining

    for code in iterator:
        if not HAS_TQDM:
            print(f"  [{len(done)+1}/{len(qual_codes)}] {code} ...", end=" ", flush=True)

        row: dict = {"Qualification_Code": code}

        # Taxonomy
        tax = scrape_taxonomy(code)
        row.update(tax)
        time.sleep(SLEEP_SECONDS)

        # Units
        units = scrape_units(code)
        row["Core_Units_Count"]     = len(units["core_units"])
        row["Elective_Units_Count"] = len(units["elective_units"])
        row["Core_Units"]           = units_to_string(units["core_units"])
        row["Elective_Units"]       = units_to_string(
                                          units["elective_units"],
                                          include_group=True
                                      )
        # Pipe-separated codes only (useful for lookups)
        row["Core_Unit_Codes"]     = "|".join(u["unit_code"] for u in units["core_units"])
        row["Elective_Unit_Codes"] = "|".join(u["unit_code"] for u in units["elective_units"])

        done[code] = row

        if not HAS_TQDM:
            print("done.")

        # Save checkpoint after every record
        pd.DataFrame(list(done.values())).to_csv(CHECKPOINT, index=False)
        time.sleep(SLEEP_SECONDS)

    # ── Merge and export ──────────────────────────────────────────────────────
    scraped_df = pd.DataFrame(list(done.values()))

    # Merge on qualification code
    merged = df.merge(
        scraped_df,
        left_on  = "Qualification Code",
        right_on = "Qualification_Code",
        how      = "left"
    ).drop(columns=["Qualification_Code"], errors="ignore")

    # Reorder: original columns first, then new scraped columns
    original_cols = df.columns.tolist()
    new_cols = [c for c in merged.columns if c not in original_cols]
    merged   = merged[original_cols + new_cols]

    merged.to_excel(output_path, index=False)
    print(f"\n✅  Saved to: {output_path}")
    print(f"   Rows: {len(merged)}   |   Columns: {len(merged.columns)}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape TGA taxonomy + units for all qualifications.")
    parser.add_argument(
        "--input",  "-i",
        default="data__57_.xlsx",
        help="Source Excel file (default: data__57_.xlsx)"
    )
    parser.add_argument(
        "--output", "-o",
        default="tga_qualifications_updated.xlsx",
        help="Output Excel file (default: tga_qualifications_updated.xlsx)"
    )
    args = parser.parse_args()
    main(args.input, args.output)
