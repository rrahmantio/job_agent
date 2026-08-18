"""
APAC AI Career Agent — weekly Australia + Singapore career radar.

Searches for specific, currently relevant AI/data/cloud/solutions/GTM roles,
scores them against the candidate profile, prioritizes visa sponsorship, and
emails the top 20 opportunities each week.

Required environment variables:
    SERPER_API_KEY
    OPENAI_API_KEY
    EMAIL_TO
    SMTP_USER
    SMTP_PASSWORD

Optional:
    SMTP_HOST (default: smtp.gmail.com)
    SMTP_PORT (default: 465)
    MIN_SCORE (default: 72)
    MAX_RESULTS_PER_QUERY (default: 10)
"""

import json
import os
import re
import smtplib
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from html import escape
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "seen_jobs.sqlite3"

SERPER_URL = "https://google.serper.dev/search"
OPENAI_MODEL = "gpt-4.1-mini"

MAX_EMAIL_JOBS = 20
FETCH_WORKERS = 8
SCORE_WORKERS = 6
MAX_POSTING_AGE_DAYS = 30

PROFILE = """
Candidate based in Jakarta, Indonesia, currently a Presales Manager at CKDelta
(part of Indosat Ooredoo Hutchison Group).

Core profile:
- Enterprise AI, data and digital transformation presales / solutions consulting
- Enterprise GTM, discovery, solution design, demos, POCs/PoVs, RFP/RFI,
  executive/C-level engagement and strategic account management
- AI/ML, predictive analytics, GenAI, RAG, agentic AI, AI architecture,
  enterprise data foundation, Databricks / Delta Lake
- Cloud / integration familiarity: GCP, AWS/Azure concepts, APIs, Kafka,
  Kubernetes, Docker, SFTP, SAP/Oracle integration
- Previous SaaS / enterprise technology experience at Zoho, Telesign and Vymo
- Indonesia / ASEAN enterprise exposure across large enterprises, BUMN,
  utilities, energy, manufacturing, insurance and financial services
- AI use cases involving utilities, energy, industrial AI, computer vision,
  predictive maintenance, customer AI agents and regulatory intelligence
- Target trajectory: Enterprise AI Solutions, AI Deployment / AI Transformation,
  Solutions Consulting, Solutions Architecture, or regional AI/data/cloud GTM
- Target geography: Australia and Singapore
- Candidate does NOT currently have local work rights in Australia or Singapore
  and would require employer-sponsored work authorization / relocation.
- Strong preference for mid-size tech/AI/data/cloud/cyber companies,
  growth-stage firms and selected startups; avoid generic junior sales roles.
"""

QUERIES = [
    # Australia
    'site:linkedin.com/jobs Australia ("Solutions Consultant" OR "Solutions Engineer" OR "Solutions Architect") (AI OR data OR cloud OR analytics)',
    'site:linkedin.com/jobs Australia ("AI Deployment" OR "AI Transformation" OR "AI Strategy" OR "AI Consultant")',
    'site:linkedin.com/jobs Australia ("Principal Consultant" OR "Principal Solution Architect") (AI OR data OR analytics OR cloud)',
    'site:linkedin.com/jobs Australia ("Customer Solutions" OR "Technical Consultant" OR "GTM") (AI OR data OR cloud)',
    'site:linkedin.com/jobs Australia (Databricks OR "Generative AI" OR "AI agents") ("solutions" OR "consultant" OR "presales")',
    # Singapore
    'site:linkedin.com/jobs Singapore ("Solutions Consultant" OR "Solutions Engineer" OR "Solutions Architect") (AI OR data OR cloud OR analytics)',
    'site:linkedin.com/jobs Singapore ("AI Deployment" OR "AI Transformation" OR "AI Strategy" OR "AI Consultant")',
    'site:linkedin.com/jobs Singapore ("Principal Consultant" OR "Principal Solution Architect") (AI OR data OR analytics OR cloud)',
    'site:linkedin.com/jobs Singapore ("Customer Solutions" OR "Technical Consultant" OR "GTM") (AI OR data OR cloud)',
    'site:linkedin.com/jobs Singapore (Databricks OR "Generative AI" OR "AI agents") ("solutions" OR "consultant" OR "presales")',
    # Explicit sponsorship discovery
    'Australia AI "solutions consultant" "visa sponsorship" job',
    'Australia "data consultant" "visa sponsorship" AI job',
    'Australia "solutions engineer" "visa sponsorship" AI job',
    'Singapore "solutions consultant" "visa sponsorship" AI job',
    'Singapore "solutions engineer" "visa sponsorship" AI job',
]


@dataclass
class Job:
    title: str
    company: str
    location: str
    url: str
    snippet: str
    source_query: str
    score: int = 0
    tier: str = ""
    visa: str = ""
    rationale: str = ""
    cv_tweak: str = ""
    status: str = ""
    specific: bool = False
    company_size: str = ""
    previous_score: int | None = None
    previous_rank: int | None = None
    current_rank: int | None = None
    label: str = ""
    posted_at: datetime | None = None
    posted_date_text: str = ""
    freshness_days: int | None = None
    source_company: str = ""


def env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name)
    if not value:
        value = default
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value or ""


def expected_country(query: str) -> str:
    return "Australia" if "Australia" in query else "Singapore"


def parse_posting_date(value: str, now: datetime | None = None) -> tuple[datetime | None, str]:
    """Parse common Google/ATS relative and absolute posting-date strings."""
    if not value:
        return None, ""
    now = now or datetime.now(timezone.utc)
    text = re.sub(r"\s+", " ", str(value)).strip()
    lower = text.lower()

    # Relative forms commonly returned by Google/Serper.
    m = re.search(r"(\d+)\s*(minute|hour|day|week|month|year)s?\s+ago", lower)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        days = {
            "minute": 0, "hour": 0, "day": 1, "week": 7,
            "month": 30, "year": 365,
        }[unit]
        return now - timedelta(days=n * days), text

    if "yesterday" in lower:
        return now - timedelta(days=1), text
    if "today" in lower or "just posted" in lower or "just now" in lower:
        return now, text

    # ISO timestamps and common date formats.
    iso = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc), text
    except ValueError:
        pass

    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return dt, text
        except ValueError:
            continue

    return None, text


def extract_posting_date(text: str, now: datetime | None = None) -> tuple[datetime | None, str]:
    """Extract a likely posting date from SERP snippets/page metadata."""
    if not text:
        return None, ""
    now = now or datetime.now(timezone.utc)

    patterns = [
        r"\b\d+\s*(?:minute|hour|day|week|month|year)s?\s+ago\b",
        r"\byesterday\b",
        r"\btoday\b",
        r"\bjust posted\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            dt, raw = parse_posting_date(match.group(0), now)
            if dt:
                return dt, raw
    return None, ""


def set_posting_date(job: Job, raw_date: str, extra_text: str = "") -> None:
    now = datetime.now(timezone.utc)
    dt, text = parse_posting_date(raw_date, now)
    if dt is None:
        dt, text = extract_posting_date(extra_text, now)
    if dt is not None:
        job.posted_at = dt
        job.posted_date_text = text or raw_date
        job.freshness_days = max(0, (now - dt).days)


def freshness_bucket(job: Job) -> int:
    """Higher is fresher; unknown dates are deliberately lowest."""
    if job.freshness_days is None:
        return 0
    if job.freshness_days <= 3:
        return 6
    if job.freshness_days <= 7:
        return 5
    if job.freshness_days <= 14:
        return 4
    if job.freshness_days <= 30:
        return 3
    if job.freshness_days <= 45:
        return 1
    return -5


def freshness_score(job: Job) -> int:
    """Use freshness as a major ranking signal without overpowering sponsorship."""
    if job.freshness_days is None:
        return 0
    # 40 points for today, declining to 0 at 30 days, then a penalty.
    if job.freshness_days <= 30:
        return max(0, 40 - job.freshness_days)
    if job.freshness_days <= 45:
        return 5 - (job.freshness_days - 30)
    return -20


def ranking_score(job: Job) -> int:
    # Fit + visa remain important, but freshness is now a first-class signal.
    return priority_score(job) + freshness_score(job)


def search_serper(query: str, num: int = 10) -> list[dict[str, Any]]:
    country = expected_country(query)

    response = requests.post(
        SERPER_URL,
        headers={
            "X-API-KEY": env("SERPER_API_KEY", required=True),
            "Content-Type": "application/json",
        },
        json={
            "q": query,
            "num": num,
            "gl": "au" if country == "Australia" else "sg",
            "hl": "en",
            # Keep discovery focused on jobs likely to have changed since the
            # previous weekly run. We still rank by exact posting date below.
            "tbs": "qdr:m",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("organic", [])


def normalize_url(url: str) -> str:
    return (url or "").split("#")[0].split("?")[0].rstrip("/")


def is_linkedin(url: str) -> bool:
    return "linkedin.com/" in url.lower()


def is_specific_linkedin_job(url: str) -> bool:
    url_lower = url.lower()
    return (
        "linkedin.com/jobs/view/" in url_lower
        or "linkedin.com/comm/jobs/view/" in url_lower
    )


def extract_linkedin_employer(search_title: str) -> str:
    """
    Extract the employer from common LinkedIn SERP title formats.

    Examples:
      "Workiva hiring Solutions Architect in Singapore, Singapore | LinkedIn"
      "Microsoft hiring Data and AI Solution Architect in Singapore | LinkedIn"
      "Solutions Architect at Workiva | LinkedIn"
    """
    text = re.sub(r"\s+", " ", (search_title or "")).strip()

    patterns = [
        r"^(?P<company>.+?)\s+hiring\s+.+?(?:\s+in\s+.+?)?\s*\|\s*LinkedIn\s*$",
        r"^(?P<title>.+?)\s+at\s+(?P<company>.+?)\s*\|\s*LinkedIn\s*$",
    ]

    for pattern in patterns:
        match = re.match(pattern, text, flags=re.I)
        if match:
            company = match.groupdict().get("company", "").strip()
            if company:
                return company

    # Some LinkedIn results omit the "| LinkedIn" suffix.
    match = re.match(
        r"^(?P<company>.+?)\s+hiring\s+.+?(?:\s+in\s+.+?)?$",
        text,
        flags=re.I,
    )
    if match:
        company = match.group("company").strip()
        if company:
            return company

    return ""


GENERIC_TITLE_PATTERNS = [
    r"\bvarious\b",
    r"\bmultiple companies\b",
    r"\bmultiple roles\b",
    r"\bjobs?\s+in\b",
    r"\bjob listings?\b",
    r"\bjob opportunities\b",
    r"\bsearch results?\b",
    r"\bfind jobs?\b",
    r"\bvisa sponsorship\b.*\bjobs?\b",
    r"\bjobs?\b.*\bvisa sponsorship\b",
    r"\bsolutions architect jobs?\b",
    r"\bsolutions engineer jobs?\b",
    r"\bsolutions consultant jobs?\b",
    r"\bprincipal solutions engineer jobs?\b",
    r"\bdata engineer jobs?\b",
]


def looks_like_generic_listing(title: str, url: str) -> bool:
    title_lower = (title or "").strip().lower()
    url_lower = (url or "").lower()

    if any(re.search(pattern, title_lower) for pattern in GENERIC_TITLE_PATTERNS):
        return True

    # LinkedIn search/category pages are not individual vacancies.
    if "linkedin.com/jobs/" in url_lower and not is_specific_linkedin_job(url_lower):
        return True

    generic_url_bits = (
        "/search/",
        "/search-results",
        "/job-search",
        "/jobs-in-",
        "/careers/search",
    )
    return any(bit in url_lower for bit in generic_url_bits)


def looks_closed(text: str) -> bool:
    text_lower = (text or "").lower()
    closed_phrases = [
        "no longer accepting applications",
        "applications are closed",
        "job is no longer available",
        "position has been filled",
        "this job is no longer available",
        "expired",
        "no longer accepting",
        "applications have closed",
        "applications closed",
        "role has been filled",
        "position is filled",
        "this vacancy has closed",
        "vacancy has closed",
        "job has closed",
        "job closed",
        "this position is no longer available",
    ]
    return any(phrase in text_lower for phrase in closed_phrases)


def fetch_page_text(url: str) -> str:
    # LinkedIn is intentionally not scraped. Its SERP title/snippet is used.
    if is_linkedin(url):
        return ""

    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; APAC-AI-Career-Agent/2.0)"
                )
            },
            timeout=8,
            allow_redirects=True,
        )
        if response.status_code >= 400:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        return re.sub(
            r"\s+",
            " ",
            soup.get_text(" ", strip=True),
        )[:12000]
    except Exception:
        return ""


def discover_jobs() -> list[Job]:
    found: dict[str, Job] = {}
    max_results = int(env("MAX_RESULTS_PER_QUERY", "10"))

    print(f"Running {len(QUERIES)} search queries...")

    for index, query in enumerate(QUERIES, 1):
        try:
            results = search_serper(query, max_results)
            print(f"  Search {index}/{len(QUERIES)}: {len(results)} results")
        except Exception as exc:
            print(f"[WARN] Search failed: {query}\n       {exc}")
            continue

        country = expected_country(query)

        for item in results:
            url = normalize_url(item.get("link", ""))
            title = (item.get("title", "") or "").strip()
            snippet = (item.get("snippet", "") or "").strip()

            if not url or not title:
                continue

            # Reject obvious "Various / jobs / search results" pages before
            # spending time or OpenAI tokens on them.
            if looks_like_generic_listing(title, url):
                continue

            job = found.setdefault(
                url.lower(),
                Job(
                    title=title,
                    company="Unknown",
                    location=country,
                    url=url,
                    snippet=snippet,
                    source_query=query,
                    source_company=(
                        extract_linkedin_employer(title)
                        if is_linkedin(url)
                        else ""
                    ),
                ),
            )
            # Serper/Google frequently exposes a relative posting date in the
            # result. Keep it before page enrichment because LinkedIn is not
            # scraped by this agent.
            result_date = str(item.get("date", "") or "").strip()
            set_posting_date(job, result_date, f"{title} {snippet}")

    jobs = list(found.values())
    print(
        f"Collected {len(jobs)} specific-looking job URLs "
        "before page enrichment."
    )

    # Fetch non-LinkedIn pages concurrently. This removes the previous
    # ~15-seconds-per-page bottleneck.
    fetch_jobs = [job for job in jobs if not is_linkedin(job.url)]
    page_text: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        future_map = {
            executor.submit(fetch_page_text, job.url): job.url
            for job in fetch_jobs
        }

        for future in as_completed(future_map):
            url = future_map[future]
            try:
                page_text[url] = future.result()
            except Exception:
                page_text[url] = ""

    valid: list[Job] = []

    for job in jobs:
        page = page_text.get(job.url, "")
        combined = f"{job.title}\n{job.snippet}\n{page}"

        if looks_closed(combined):
            continue

        if page:
            job.snippet = (job.snippet + "\n" + page[:7000])[:9000]
            if job.posted_at is None:
                set_posting_date(job, "", page[:9000])

        # A weekly radar should not keep resurfacing genuinely old vacancies.
        # Unknown dates are retained (but ranked below dated openings) because
        # some ATS pages do not expose structured posting dates.
        if job.freshness_days is not None and job.freshness_days > MAX_POSTING_AGE_DAYS:
            continue

        valid.append(job)

    return valid


SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "title": {"type": "string"},
        "location": {"type": "string"},
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "tier": {"type": "string"},
        "visa": {"type": "string"},
        "is_target": {"type": "boolean"},
        "is_specific_posting": {"type": "boolean"},
        "company_size": {"type": "string"},
        "rationale": {"type": "string"},
        "cv_tweak": {"type": "string"},
    },
    "required": [
        "company",
        "title",
        "location",
        "score",
        "tier",
        "visa",
        "is_target",
        "is_specific_posting",
        "company_size",
        "rationale",
        "cv_tweak",
    ],
    "additionalProperties": False,
}


def score_job(client: OpenAI, job: Job) -> Job:
    prompt = f"""
You are an executive career consultant evaluating ONE job for this candidate.

CANDIDATE:
{PROFILE}

JOB SOURCE:
Title from search: {job.title}
URL: {job.url}
Expected country from search: {job.location}
Search snippet / page text:
{job.snippet}

IMPORTANT POSTING-QUALITY RULES:
- This must be ONE specific open role at ONE specific employer.
- Reject search-result pages, category pages, job collections, recruitment
  search pages, "Various" listings, "Multiple companies" pages, generic
  "Solutions Architect jobs" pages, generic visa-sponsorship job pages, and
  any page that does not identify one actual vacancy.
- company must be the ACTUAL EMPLOYER.
- title must be the ACTUAL JOB TITLE.
- NEVER infer the employer from a company merely mentioned in the job
  description, page text, technology stack, customer list, or search snippet.
- For LinkedIn results, the employer shown in the LinkedIn result title is
  authoritative. If the result says "Workiva hiring Solutions Architect",
  the employer is Workiva even if the page text mentions Anthropic, OpenAI,
  Microsoft, AWS, or another company.
- Never invent a company or title from examples mentioned on an aggregator page.
- If one specific company + one specific role cannot be confidently identified,
  set is_specific_posting=false and is_target=false.

TARGETING:
1. Australia or Singapore only.
2. Tier A or Tier B companies only. Reject Tier C / low-quality / obviously
   irrelevant employers.
3. Prefer mid-size tech/AI/data/cloud/cyber/growth companies and strong
   specialist technology companies. Big Tech can be considered only if the
   role is unusually strong for this candidate.
4. Strongly prefer Solutions Consultant, Solutions Engineer, Solutions Architect,
   Principal Consultant, AI Deployment, AI Transformation, AI Strategy,
   Customer Solutions, Enterprise AI, technical GTM, data/cloud consulting,
   and adjacent senior customer-facing technical roles.
5. Reject junior roles, internships, pure software engineering, pure data
   engineering, generic SDR/BDR roles, and roles requiring deep engineering
   credentials the candidate clearly does not have.

VISA:
- Candidate currently lacks local work rights in Australia/Singapore and needs
  employer-sponsored work authorization / relocation.
- Explicit sponsorship, relocation support, international applicants, or clear
  employer sponsorship language = positive.
- "Must have unrestricted Australian/Singapore work rights", "no sponsorship",
  or equivalent = reject.
- Do NOT infer sponsorship merely because the employer is multinational.
- Use exactly one practical visa label:
  "🟢🟢 Jackpot" = unusually clear sponsorship/relocation evidence
  "🟢 Excellent" = explicit or strong evidence of sponsorship
  "🟡 Possible / verify" = not stated or uncertain
  "🔴 No" = clearly incompatible; set is_target=false
- We do not email red/no-visa roles.

POSTING RECENCY:
- Prefer roles with a clearly recent posting date. The agent separately ranks
  jobs by posting recency; do not invent a date if the source does not provide one.
- Treat an explicit closed/expired/filled signal as disqualifying.

SCORING:
Score 0-100:
- 45% CV/role fit
- 25% career trajectory toward Enterprise AI
- 20% visa feasibility
- 10% company/market attractiveness
Only set is_target=true if the role is genuinely strong and score >= 72.
Tier must be exactly "A" or "B" for a target.

Return a concise rationale and 2-4 concrete CV tweaks.
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "job_match",
                "strict": True,
                "schema": SCHEMA,
            }
        },
    )

    data = json.loads(response.output_text)

    model_company = data["company"].strip()
    source_company = job.source_company.strip()

    if source_company:
        if (
            model_company
            and re.sub(r"\W+", "", model_company.lower())
            != re.sub(r"\W+", "", source_company.lower())
        ):
            print(
                f"[WARN] Employer mismatch for {job.url}: "
                f"model={model_company!r}, source={source_company!r}. "
                "Using source employer."
            )
        job.company = source_company
    else:
        job.company = model_company

    job.title = data["title"].strip()
    job.location = data["location"].strip()
    job.score = int(data["score"])
    job.tier = data["tier"].strip().upper()
    job.visa = data["visa"].strip()
    job.company_size = data["company_size"].strip()
    job.rationale = data["rationale"].strip()
    job.cv_tweak = data["cv_tweak"].strip()
    job.specific = bool(data["is_specific_posting"])
    job.status = "target" if data["is_target"] else "reject"

    if not job.specific:
        job.status = "reject"

    if job.tier not in {"A", "B"}:
        job.status = "reject"

    if job.visa.startswith("🔴"):
        job.status = "reject"

    return job


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_jobs (
                url TEXT PRIMARY KEY,
                title TEXT,
                company TEXT,
                first_seen TEXT,
                last_seen TEXT,
                score INTEGER,
                last_rank INTEGER,
                visa TEXT
            )
            """
        )

        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(seen_jobs)")
        }

        if "last_rank" not in existing_columns:
            conn.execute(
                "ALTER TABLE seen_jobs ADD COLUMN last_rank INTEGER"
            )

        if "visa" not in existing_columns:
            conn.execute(
                "ALTER TABLE seen_jobs ADD COLUMN visa TEXT"
            )

        conn.commit()


def get_history(url: str) -> tuple[int | None, int | None]:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT score, last_rank FROM seen_jobs WHERE url=?",
            (url,),
        ).fetchone()

    if not row:
        return None, None

    return row[0], row[1]


def save_seen(job: Job, rank: int | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO seen_jobs(
                url, title, company, first_seen, last_seen,
                score, last_rank, visa
            )
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(url) DO UPDATE SET
                title=excluded.title,
                company=excluded.company,
                last_seen=excluded.last_seen,
                score=excluded.score,
                last_rank=excluded.last_rank,
                visa=excluded.visa
            """,
            (
                job.url,
                job.title,
                job.company,
                now,
                now,
                job.score,
                rank,
                job.visa,
            ),
        )
        conn.commit()


def visa_priority(visa: str) -> int:
    visa_lower = visa.lower()

    if "jackpot" in visa_lower:
        return 12

    if "excellent" in visa_lower:
        return 8

    return 0


def priority_score(job: Job) -> int:
    return job.score + visa_priority(job.visa)


def country_of(job: Job) -> str:
    text = f"{job.location} {job.source_query}".lower()

    if "singapore" in text:
        return "Singapore"

    if any(
        word in text
        for word in (
            "australia",
            "melbourne",
            "sydney",
            "brisbane",
            "adelaide",
            "perth",
        )
    ):
        return "Australia"

    return ""


def dedupe_scored(jobs: list[Job]) -> list[Job]:
    best: dict[tuple[str, str, str], Job] = {}

    for job in jobs:
        key = (
            re.sub(r"\W+", "", job.company.lower()),
            re.sub(r"\W+", "", job.title.lower()),
            country_of(job).lower(),
        )

        current = best.get(key)

        if current is None or priority_score(job) > priority_score(current):
            best[key] = job

    return list(best.values())


def select_top_20(jobs: list[Job]) -> list[Job]:
    """
    Select the 20 strongest CURRENT openings.

    Freshness is now a first-class ranking signal. The practical priority is:
    1) explicit sponsorship strength, 2) CV/company fit, 3) posting recency.
    Among otherwise similar roles, the newest opening wins decisively.
    Roles older than MAX_POSTING_AGE_DAYS days are removed during discovery.
    """
    australia = sorted(
        [j for j in jobs if country_of(j) == "Australia"],
        key=ranking_score,
        reverse=True,
    )
    singapore = sorted(
        [j for j in jobs if country_of(j) == "Singapore"],
        key=ranking_score,
        reverse=True,
    )

    sponsored = [
        job for job in jobs
        if "jackpot" in job.visa.lower()
        or "excellent" in job.visa.lower()
    ]
    sponsored.sort(key=ranking_score, reverse=True)

    if len(sponsored) >= MAX_EMAIL_JOBS:
        selected = sponsored[:MAX_EMAIL_JOBS]
    else:
        selected = list(sponsored)
        selected_urls = {job.url for job in selected}
        remaining_au = [j for j in australia if j.url not in selected_urls]
        remaining_sg = [j for j in singapore if j.url not in selected_urls]
        slots = MAX_EMAIL_JOBS - len(selected)

        au_slots = slots // 2
        sg_slots = slots - au_slots
        selected.extend(remaining_au[:au_slots])
        selected_urls = {job.url for job in selected}
        selected.extend(
            [j for j in remaining_sg if j.url not in selected_urls][:sg_slots]
        )

        selected_urls = {job.url for job in selected}
        leftovers = [
            j for j in jobs
            if j.url not in selected_urls
            and country_of(j) in {"Australia", "Singapore"}
        ]
        leftovers.sort(key=ranking_score, reverse=True)
        selected.extend(leftovers[: MAX_EMAIL_JOBS - len(selected)])

    # FINAL ORDER: newest opening first, with fit/sponsorship breaking ties.
    # Unknown dates go after dated openings so they don't crowd out fresh jobs.
    return sorted(
        selected,
        key=lambda j: (
            j.posted_at is not None,
            j.posted_at or datetime.min.replace(tzinfo=timezone.utc),
            priority_score(j),
        ),
        reverse=True,
    )[:MAX_EMAIL_JOBS]


def score_jobs(
    client: OpenAI,
    jobs: list[Job],
    min_score: int,
) -> list[Job]:
    results: list[Job] = []

    print(
        f"Scoring {len(jobs)} candidates with "
        f"{SCORE_WORKERS} parallel workers..."
    )

    with ThreadPoolExecutor(max_workers=SCORE_WORKERS) as executor:
        future_map = {
            executor.submit(score_job, client, job): job
            for job in jobs
        }

        completed = 0

        for future in as_completed(future_map):
            original = future_map[future]
            completed += 1

            try:
                scored = future.result()

                previous_score, previous_rank = get_history(scored.url)
                scored.previous_score = previous_score
                scored.previous_rank = previous_rank

                if scored.source_company:
                    scored.company = scored.source_company

                if (
                    scored.status == "target"
                    and scored.score >= min_score
                    and scored.specific
                    and scored.tier in {"A", "B"}
                    and country_of(scored) in {"Australia", "Singapore"}
                    and not scored.visa.startswith("🔴")
                ):
                    results.append(scored)

                print(
                    f"[{completed}/{len(jobs)}] "
                    f"{scored.score:>3} | "
                    f"{scored.visa:<20} | "
                    f"{scored.company} — {scored.title}"
                )

            except Exception as exc:
                print(
                    f"[WARN] Scoring failed for {original.url}: {exc}"
                )

    return dedupe_scored(results)


def classify_label(job: Job) -> str:
    if job.previous_score is None:
        return "🆕 NEW"

    if (
        job.previous_rank is not None
        and job.current_rank is not None
        and job.previous_rank > job.current_rank
    ):
        return "📈 RISING"

    if job.score >= (job.previous_score or 0) + 5:
        return "📈 RISING"

    return "↔️ STILL STRONG"


def send_email(jobs: list[Job]) -> None:
    if not jobs:
        print("No qualifying jobs found. No email sent.")
        return

    today = datetime.now().strftime("%d %b %Y")

    australia_count = sum(
        country_of(job) == "Australia" for job in jobs
    )
    singapore_count = sum(
        country_of(job) == "Singapore" for job in jobs
    )

    html_parts = [
        f"<h2>🎯 APAC AI Career Radar — {today}</h2>",
        (
            f"<p><b>{len(jobs)} highest-priority roles</b> | "
            f"🇦🇺 Australia: {australia_count} | "
            f"🇸🇬 Singapore: {singapore_count}</p>"
        ),
        (
            "<p>Ranked by CV fit, Enterprise AI trajectory, visa feasibility "
            "and company quality. Specific employer + specific role only.</p>"
        ),
    ]

    for rank, job in enumerate(jobs, 1):
        country = country_of(job)

        html_parts.append(
            f"""
            <hr>
            <h3>
                #{rank} — {escape(job.company)}
                — {escape(job.title)} — {job.score}/100
            </h3>
            <p>
                <b>{escape(country)}</b>
                · Tier {escape(job.tier)}
                · {escape(job.company_size)}
                <br>
                {escape(job.label)}
                · {escape(job.visa)}
                · {escape(job.posted_date_text or "Posting date not exposed")}
            </p>
            <p>
                <b>Why it fits:</b> {escape(job.rationale)}
            </p>
            <p>
                <b>CV tweak:</b> {escape(job.cv_tweak)}
            </p>
            <p>
                <a href="{escape(job.url, quote=True)}">
                    Open job posting →
                </a>
            </p>
            """
        )

    html = "\n".join(html_parts)
    text = re.sub(r"<[^>]+>", "", html)
    text = text.replace("&nbsp;", " ")

    message = EmailMessage()
    message["Subject"] = (
        f"🎯 APAC AI Jobs — Top {len(jobs)} | "
        f"AU {australia_count} / SG {singapore_count}"
    )
    message["From"] = env("SMTP_USER", required=True)
    message["To"] = env("EMAIL_TO", required=True)

    message.set_content(text)
    message.add_alternative(html, subtype="html")

    host = env("SMTP_HOST", "smtp.gmail.com")
    port = int(env("SMTP_PORT", "465"))

    with smtplib.SMTP_SSL(host, port) as smtp:
        smtp.login(
            env("SMTP_USER", required=True),
            env("SMTP_PASSWORD", required=True),
        )
        smtp.send_message(message)

    print(f"Email sent with {len(jobs)} jobs.")


def main() -> None:
    init_db()

    min_score = int(env("MIN_SCORE", "72"))

    print("Discovering jobs...")
    jobs = discover_jobs()

    dated_jobs = sum(job.posted_at is not None for job in jobs)
    print(
        f"Discovered {len(jobs)} non-obviously-closed specific-looking candidates "
        f"({dated_jobs} with posting dates)."
    )

    if not jobs:
        print("No candidates discovered.")
        return

    client = OpenAI(
        api_key=env("OPENAI_API_KEY", required=True)
    )

    qualifying = score_jobs(client, jobs, min_score)

    print(
        f"Qualifying Tier A/B candidates: {len(qualifying)}"
    )

    selected = select_top_20(qualifying)

    for rank, job in enumerate(selected, 1):
        job.current_rank = rank
        job.label = classify_label(job)

    selected_urls = {job.url for job in selected}

    # Persist all qualifying jobs, not just the emailed 20, so the next weekly
    # run can identify NEW/RISING opportunities.
    for job in qualifying:
        rank = (
            selected.index(job) + 1
            if job.url in selected_urls
            else None
        )
        save_seen(job, rank)

    australia_count = sum(
        country_of(job) == "Australia" for job in selected
    )
    singapore_count = sum(
        country_of(job) == "Singapore" for job in selected
    )

    print(
        f"Selected {len(selected)} jobs for email: "
        f"{australia_count} Australia / {singapore_count} Singapore."
    )

    send_email(selected)


if __name__ == "__main__":
    main()
