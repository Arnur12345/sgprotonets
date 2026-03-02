import re
import json
import subprocess


def fetch_form_html_via_chrome(form_url: str) -> str:
    """
    Uses AppleScript to open the URL in Chrome (already logged into Google),
    waits for the page to load, then extracts the page source.
    """
    ascript = f'''
    tell application "Google Chrome"
        activate
        if (count of windows) is 0 then
            make new window
        end if
        set newTab to make new tab at end of tabs of front window
        set URL of newTab to "{form_url}"
        delay 7
        set pageSource to execute newTab javascript "document.documentElement.outerHTML"
        close newTab
        return pageSource
    end tell
    '''

    result = subprocess.run(
        ["osascript", "-e", ascript],
        capture_output=True, text=True, timeout=30
    )

    if result.returncode != 0:
        raise RuntimeError(f"AppleScript error: {result.stderr.strip()}")

    return result.stdout


def fetch_form_data_via_js(form_url: str):
    """
    Loads the form in Chrome and extracts FB_PUBLIC_LOAD_DATA_ directly via JS.
    This captures ALL pages/sections of a multi-page form at once.
    """
    ascript = f'''
    tell application "Google Chrome"
        activate
        if (count of windows) is 0 then
            make new window
        end if
        set newTab to make new tab at end of tabs of front window
        set URL of newTab to "{form_url}"
        delay 7
        set formData to execute newTab javascript "JSON.stringify(typeof FB_PUBLIC_LOAD_DATA_ !== 'undefined' ? FB_PUBLIC_LOAD_DATA_ : null)"
        close newTab
        return formData
    end tell
    '''
    result = subprocess.run(
        ["osascript", "-e", ascript],
        capture_output=True, text=True, timeout=30
    )

    if result.returncode != 0:
        raise RuntimeError(f"AppleScript error: {result.stderr.strip()}")

    output = result.stdout.strip()
    if not output or output == "null":
        return None

    return json.loads(output)


def extract_google_form_questions(form_url: str):
    """
    Fetches a Google Form via the user's authenticated Chrome session
    and extracts all questions + answer variants from ALL pages/sections.
    """
    # Try JS extraction first (gets all data including multi-page)
    raw_data = fetch_form_data_via_js(form_url)

    if not raw_data:
        # Fallback: try HTML source parsing
        html = fetch_form_html_via_chrome(form_url)
        match = re.search(
            r"FB_PUBLIC_LOAD_DATA_\s*=\s*(\[.*?\]);\s*</script>",
            html, re.DOTALL
        )
        if not match:
            with open("form_debug.html", "w", encoding="utf-8") as f:
                f.write(html)
            raise ValueError(
                "Could not extract form data. Raw HTML saved to form_debug.html"
            )
        raw_data = json.loads(match.group(1))

    # Save raw data for debugging
    with open("form_raw_data.json", "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)

    return _parse_form_data(raw_data)


def _parse_form_data(raw_data):
    """Parse the FB_PUBLIC_LOAD_DATA_ JSON structure (all pages)."""
    form_title = raw_data[1][8]
    form_desc  = raw_data[1][0]
    questions_raw = raw_data[1][1]

    # Page/section info is embedded in the question list.
    # Items with item[4] == None and a title are section headers (page breaks).
    questions = []
    current_page = 1

    for item in questions_raw:
        question_title = item[1] if len(item) > 1 else ""
        question_desc  = item[2] if len(item) > 2 else ""
        type_block     = item[4] if len(item) > 4 else None

        # Section header / page break (no type_block, but has a title)
        if type_block is None:
            if question_title:
                current_page += 1
                questions.append({
                    "page":        current_page,
                    "title":       question_title,
                    "description": question_desc or "",
                    "type":        "Section header",
                    "variants":    [],
                    "required":    False,
                })
            continue

        # Question type is at item[3]
        raw_type = item[3] if len(item) > 3 else None

        type_map = {
            0:  "Short answer",
            1:  "Paragraph",
            2:  "Multiple choice",
            3:  "Checkboxes",
            4:  "Dropdown",
            5:  "Linear scale",
            7:  "Multiple choice grid",
            8:  "Checkbox grid",
            9:  "Date",
            10: "Time",
            13: "File upload",
            18: "Rating scale",
        }
        question_type = type_map.get(raw_type, f"Unknown ({raw_type})")

        # First sub-block has options and required flag
        sub_block = type_block[0] if type_block else None
        variants = []
        required = False

        if sub_block:
            # Options/variants at sub_block[1]
            options_block = sub_block[1] if len(sub_block) > 1 and sub_block[1] else []
            if options_block:
                for opt in options_block:
                    if opt and opt[0]:
                        variants.append(opt[0])

            # Required flag at sub_block[2]
            required = bool(sub_block[2]) if len(sub_block) > 2 else False

            # Linear scale: extract low/high labels
            if raw_type == 5 and len(sub_block) > 3 and sub_block[3]:
                scale_labels = sub_block[3]
                low_label  = scale_labels[0] if len(scale_labels) > 0 else ""
                high_label = scale_labels[1] if len(scale_labels) > 1 else ""
                if low_label or high_label:
                    labeled = []
                    for idx, v in enumerate(variants):
                        if idx == 0 and low_label:
                            labeled.append(f"{v} ({low_label})")
                        elif idx == len(variants) - 1 and high_label:
                            labeled.append(f"{v} ({high_label})")
                        else:
                            labeled.append(v)
                    variants = labeled

        q_entry = {
            "page":        current_page,
            "title":       question_title,
            "description": question_desc or "",
            "type":        question_type,
            "variants":    variants,
            "required":    required,
        }

        # Grid questions: rows are in type_block[1:]
        if raw_type in (7, 8) and len(type_block) > 1:
            rows = []
            for row_block in type_block[1:]:
                if row_block and len(row_block) > 3 and row_block[3]:
                    row_title = row_block[3][0] if row_block[3] and row_block[3][0] else ""
                    rows.append(row_title)
            q_entry["grid_rows"] = rows

        questions.append(q_entry)

    return {
        "form_title":   form_title,
        "form_desc":    form_desc,
        "total_pages":  current_page,
        "questions":    questions,
    }


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    url = "https://forms.gle/Cs6onTXWcnz9Xrro7"
    data = extract_google_form_questions(url)

    print(f"\n📋  Form: {data['form_title']}")
    if data["form_desc"]:
        print(f"    {data['form_desc']}")
    print(f"    Pages: {data['total_pages']}")
    print("=" * 60)

    current_page = 1
    for i, q in enumerate(data["questions"], 1):
        if q["page"] != current_page:
            current_page = q["page"]
            print(f"\n{'─' * 60}")
            print(f"  PAGE {current_page}")
            print(f"{'─' * 60}")

        req = " *" if q.get("required") else ""
        if q["type"] == "Section header":
            print(f"\n── {q['title']} ──")
            if q["description"]:
                print(f"   {q['description']}")
        else:
            print(f"\nQ{i}. [{q['type']}]{req} {q['title']}")
            if q["description"]:
                print(f"     ℹ️  {q['description']}")
            if q.get("grid_rows"):
                print(f"     Columns: {', '.join(q['variants'])}")
                for r in q["grid_rows"]:
                    print(f"       ▸ {r}")
            elif q["variants"]:
                for v in q["variants"]:
                    print(f"       • {v}")

    # Dump to JSON file
    with open("form_questions.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n✅  Saved to form_questions.json ({len(data['questions'])} questions)")
