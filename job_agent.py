"""
APAC AI / Data / Cloud / Solutions Career Agent

Pipeline:
1. Search the web via Serper (Google SERP API).
2. Deduplicate job results.
3. Heuristically remove obviously closed / irrelevant roles.
4. Ask an LLM to score each job against the user's career profile.
5. Persist seen jobs in SQLite so the same role is not emailed twice.
6. Email a weekly digest containing only strong NEW matches.

Designed for GitHub Actions, but also runs locally.

Required environment variables:
    SERPER_API_KEY
    OPENAI_API_KEY
    EMAIL_TO
    SMTP_USER
    SMTP_PASSWORD

Optional:
    OPENAI_MODEL (default: gpt-4.1-mini)
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
from dataclasses import dataclass
from datetime import datetime, timezone
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

PROFILE = """
Candidate based in Jakarta, Indonesia, currently a Presales Manager at CKDelta
(part of Indosat Ooredoo Hutchison Group). Core profile:

- Enterprise AI, data and digital transformation presales / solutions consulting
- Strong enterprise GTM, discovery, solution design, demos, POCs/PoVs, RFP/RFI,
  executive/C-level engagement and strategic account management
- AI/ML, predictive analytics, GenAI, RAG, agentic AI, AI architecture,
  enterprise data foundation, Databricks / Delta Lake
- Cloud / integration familiarity: GCP, AWS/Azure concepts, APIs, Kafka,
  Kubernetes, Docker, SFTP, SAP/Oracle integration
- Previous SaaS / enterprise technology experience at Zoho, Telesign and Vymo
- Strong Indonesia / ASEAN enterprise exposure, including large enterprises,
  BUMN, utilities, energy, manufacturing, insurance and financial services
- Has worked on AI use cases involving utilities, energy, industrial AI,
  computer vision, predictive maintenance, customer AI agents and regulatory
  intelligence
- Strategic career direction: Enterprise AI Solutions, AI Deployment / AI
  Transformation, Solutions Consulting, Solutions Architecture, or regional
  AI/data/cloud GTM
- Wants Singapore or Australia; currently does NOT have local work rights and
  would require employer-sponsored work authorization / relocation.
- Particularly interested in mid-size tech/AI/data/cloud/cyber companies,
  growth-stage firms and selected startups; not generic junior sales roles.
"""

# Search strategy is deliberately broader than exact job titles.
# This helps catch roles such as "AI Deployment Strategist", "Principal Consultant",
# "Customer Solutions", "AI Transformation", etc.
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


def env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name)

    # Treat an unset OR empty environment variable as missing,
    # so optional variables correctly fall back to their defaults.
    if not value:
        value = default

    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")

    return value or ""

def search_serper(query: str, num: int = 10) -> list[dict[str, Any]]:
    r = requests.post(
        SERPER_URL,
        headers={
            "X-API-KEY": env("SERPER_API_KEY", required=True),
            "Content-Type": "application/json",
        },
        json={"q": query, "num": num, "gl": "au" if "Australia" in query else "sg", "hl": "en"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("organic", [])


def normalize_url(url: str) -> str:
    url = url.split("?")[0].rstrip("/")
    return url


def looks_closed(text: str) -> bool:
    t = text.lower()
    closed = [
        "no longer accepting applications",
        "applications are closed",
        "job is no longer available",
        "position has been filled",
        "this job is no longer available",
        "expired",
    ]
    return any(x in t for x in closed)


def fetch_page_text(url: str) -> str:
    # Best-effort only. Many job sites block automated requests, so SERP snippets
    # remain a valid fallback.
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (career-alert-agent)"},
            timeout=15,
        )
        if r.status_code >= 400:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))[:12000]
    except Exception:
        return ""


def discover_jobs() -> list[Job]:
    found: dict[str, Job] = {}
    max_results = int(env("MAX_RESULTS_PER_QUERY", "10"))

    for q in QUERIES:
        try:
            results = search_serper(q, max_results)
        except Exception as e:
            print(f"[WARN] Search failed: {q}\n       {e}")
            continue

        for item in results:
            url = normalize_url(item.get("link", ""))
            if not url:
                continue

            title = item.get("title", "")
            snippet = item.get("snippet", "")
            key = url.lower()

            # Skip obvious non-job pages.
            if not any(x in (title + " " + url).lower() for x in
                       ["job", "career", "careers", "solution", "consult", "architect",
                        "engineer", "ai", "data", "cloud", "gtm", "sales"]):
                continue

            found.setdefault(
                key,
                Job(
                    title=title,
                    company="Unknown",
                    location="Australia/Singapore",
                    url=url,
                    snippet=snippet,
                    source_query=q,
                ),
            )

    # Enrich with page text, but keep the search snippet if fetching fails.
    jobs = []
    for job in found.values():
        page = fetch_page_text(job.url)
        combined = f"{job.title}\n{job.snippet}\n{page}"
        job.status = "closed" if looks_closed(combined) else "candidate"
        if page:
            # Preserve enough detail for scoring without creating giant s.
            job.snippet = (job.snippet + "\n" + page[:7000])[:9000]
        if job.status != "closed":
            jobs.append(job)

    return jobs


SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "location": {"type": "string"},
        "score": {"type": "integer"},
        "tier": {"type": "string"},
        "visa": {"type": "string"},
        "is_target": {"type": "boolean"},
        "rationale": {"type": "string"},
        "cv_tweak": {"type": "string"},
    },
    "required": [
        "company", "location", "score", "tier", "visa",
        "is_target", "rationale", "cv_tweak"
    ],
    "additionalProperties": False,
}


def score_job(client: OpenAI, job: Job) -> Job:
    prompt = f"""
You are an executive career consultant.

Candidate:
{PROFILE}

Job discovered:
TITLE: {job.title}
URL: {job.url}
SEARCH SNIPPET / PAGE TEXT:
{job.snippet}

Evaluate this job for the candidate.

Rules:
1. Target Australia or Singapore only.
2. Target mid-size tech/AI/data/cloud/cyber/growth companies and strong
   specialist technology companies. Avoid Big Tech unless the job is unusually
   relevant.
3. Strongly prefer Solutions Consultant, Solutions Engineer, Solutions Architect,
   Principal Consultant, AI Deployment, AI Transformation, Customer Solutions,
   AI Strategy, technical GTM and similar roles.
4. Reject junior roles, pure software engineering, pure data engineering,
   generic SDR/BDR roles, and roles requiring deep engineering credentials the
   candidate clearly does not have.
5. Sponsorship matters heavily. "Visa sponsorship", "relocation", "international
   applicants", or credible evidence of sponsorship is positive. "Must have
   unrestricted Australian/Singapore work rights" is a major negative.
6. Do NOT infer sponsorship merely because a company is multinational.
7. If sponsorship is not stated, label visa as "🟡 Possible / verify".
8. If explicit sponsorship/relocation is stated, label visa as
   "🟢 Excellent" or "🟢🟢 Jackpot" if especially clear.
9. Score 0-100 using:
   - 45% CV/role fit
   - 25% career trajectory toward Enterprise AI
   - 20% visa feasibility
   - 10% company/market attractiveness
10. Recommend only score >= 72.
11. Give a concise rationale and 2-4 concrete CV changes.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
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
    job.company = data["company"]
    job.location = data["location"]
    job.score = data["score"]
    job.tier = data["tier"]
    job.visa = data["visa"]
    job.rationale = data["rationale"]
    job.cv_tweak = data["cv_tweak"]
    job.status = "target" if data["is_target"] else "reject"
    return job


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS seen_jobs (
                url TEXT PRIMARY KEY,
                title TEXT,
                company TEXT,
                first_seen TEXT,
                last_seen TEXT,
                score INTEGER
            )
        """)
        conn.commit()


def is_new(url: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT 1 FROM seen_jobs WHERE url=?", (url,)).fetchone()
        return row is None


def save_seen(job: Job):
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO seen_jobs(url,title,company,first_seen,last_seen,score)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(url) DO UPDATE SET
                last_seen=excluded.last_seen,
                score=excluded.score
        """, (job.url, job.title, job.company, now, now, job.score))
        conn.commit()


def send_email(jobs: list[Job]):
    if not jobs:
        print("No new target jobs. No email sent.")
        return

    jobs.sort(key=lambda j: j.score, reverse=True)
    today = datetime.now().strftime("%d %b %Y")

    html_parts = [
        f"<h2>APAC AI Career Agent — {today}</h2>",
        f"<p>Found <b>{len(jobs)}</b> new high-fit roles in Australia/Singapore.</p>",
    ]

    for j in jobs:
        visa = escape(j.visa)
        html_parts.append(f"""
        <hr>
        <h3>#{jobs.index(j)+1} — {escape(j.company)} — {j.score}/100</h3>
        <p><b>{escape(j.title)}</b><br>{escape(j.location)}<br>{visa}</p>
        <p><b>Why it fits:</b> {escape(j.rationale)}</p>
        <p><b>CV tweak:</b> {escape(j.cv_tweak)}</p>
        <p><a href="{escape(j.url)}">Open job posting</a></p>
        """)

    html = "\n".join(html_parts)
    text = re.sub(r"<[^>]+>", "", html).replace("&nbsp;", " ")

    msg = EmailMessage()
    msg["Subject"] = f"🎯 APAC AI Jobs: {len(jobs)} new matches"
    msg["From"] = env("SMTP_USER", required=True)
    msg["To"] = env("EMAIL_TO", required=True)
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")

    host = env("SMTP_HOST", "smtp.gmail.com")
    port = int(env("SMTP_PORT", "465"))
    with smtplib.SMTP_SSL(host, port) as smtp:
        smtp.login(env("SMTP_USER", required=True), env("SMTP_PASSWORD", required=True))
        smtp.send_message(msg)

    print(f"Email sent with {len(jobs)} jobs.")


def main():
    init_db()
    print("Discovering jobs...")
    jobs = discover_jobs()
    print(f"Discovered {len(jobs)} non-obviously-closed candidates.")

    client = OpenAI(api_key=env("OPENAI_API_KEY", required=True))
    min_score = int(env("MIN_SCORE", "72"))

    new_targets = []
    for idx, job in enumerate(jobs, 1):
        try:
            job = score_job(client, job)
            print(f"[{idx}/{len(jobs)}] {job.score:>3} {job.title} | {job.company}")
            if job.status == "target" and job.score >= min_score and is_new(job.url):
                new_targets.append(job)
            save_seen(job)
        except Exception as e:
            print(f"[WARN] Scoring failed for {job.url}: {e}")

    send_email(new_targets)


if __name__ == "__main__":
    main()
