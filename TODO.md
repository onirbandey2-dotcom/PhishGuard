# TODO

- [ ] Implement requested improvements in `app/streamlit_app.py`:
  - [x] Exclude currently scanned URL from “Top 5 Similar URLs”
  - [ ] Remove duplicate “Historical Match Found” heading
  - [ ] Ensure First/Last scan timestamps formatted as “18 Jul 2026, 7:28 PM”
  - [ ] One-time initialize `history.db` from `data/phishing url data.xlsx` without duplicates
  - [ ] Use RapidFuzz only if installed; otherwise difflib fallback
  - [ ] Verify app still runs (py_compile)

