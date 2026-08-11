# APAC AI Career Agent

A weekly agent that searches Australia + Singapore for **high-fit AI/data/cloud/cyber/solutions/GTM roles**, scores them against your career profile, checks the job text for sponsorship/work-rights signals, removes obvious closed roles, deduplicates jobs, and emails a digest.

It is intentionally designed around your target profile rather than generic job alerts.

## What it searches for

Primary role families:

- Solutions Consultant
- Solutions Engineer
- Solutions Architect
- Principal Consultant
- AI Deployment / AI Transformation
- AI Strategy
- Customer Solutions
- Technical GTM
- Enterprise AI / Data / Cloud consulting

Geography:

- Australia
- Singapore

Company universe:

- Mid-size technology companies
- AI companies
- Data / cloud companies
- Cybersecurity companies
- Growth-stage companies
- Selected startups

## Why it is "agentic"

Each weekly run has multiple stages:

1. **Discover** jobs using multiple search strategies.
2. **Enrich** results with the actual job page when accessible.
3. **Filter** obvious closed/expired postings.
4. **Reason** about candidate/job fit with an LLM.
5. **Assess visa evidence** separately from CV fit.
6. **Score** the opportunity.
7. **Remember** previously seen jobs in SQLite.
8. **Notify** only when a new job crosses the score threshold.

The important design choice is that it does **not** simply search for "Solutions Consultant". It looks for adjacent titles such as AI Deployment Strategist, Principal Consultant, AI Transformation and Customer Solutions.

## Setup

### 1. Create a Serper account

Serper provides Google search results through an API.

Create an API key and add it to GitHub Secrets as:

`SERPER_API_KEY`

### 2. Create an OpenAI API key

Add it as:

`OPENAI_API_KEY`

Optional model override:

`OPENAI_MODEL`

If omitted, the script defaults to `gpt-4.1-mini`.

### 3. Set up the destination email

The easiest option is Gmail.

Use:

- `SMTP_USER` = your Gmail address
- `SMTP_PASSWORD` = a Gmail **App Password**, not your normal Gmail password
- `EMAIL_TO` = the email address that should receive alerts

If using another SMTP provider, change `SMTP_HOST` and `SMTP_PORT`.

### 4. Upload to GitHub

Create a private repository and upload:

- `job_agent.py`
- `requirements.txt`
- `.gitignore`
- `.github/workflows/weekly-job-alert.yml`

### 5. Add GitHub Secrets

Repository → Settings → Secrets and variables → Actions → New repository secret.

Add:

- `SERPER_API_KEY`
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (optional)
- `EMAIL_TO`
- `SMTP_USER`
- `SMTP_PASSWORD`

### 6. Test it manually

GitHub → Actions → Weekly APAC AI Job Alert → Run workflow.

You should receive an email if there are new matches.

## Important: LinkedIn

The agent can discover LinkedIn job pages through search results, but it does **not** scrape or log into LinkedIn. This is intentional.

The search also looks beyond LinkedIn because many of the best roles are posted on:

- Greenhouse
- Lever
- Ashby
- company career sites
- specialist job boards

That is important because a role like the Mantel or Altis one can disappear from LinkedIn while the company's own application page remains live.

## Recommended tuning

Start with:

`MIN_SCORE=72`

After a few weeks:

- Too many results → 78 or 80
- Too few results → 65 or 68

For you, I would keep the visa signal heavily weighted.

## Next upgrade

The most valuable future improvement would be a **company intelligence layer**:

- maintain a watchlist of 50-100 target companies
- visit each company's career page every week
- detect newly opened roles
- search for sponsorship evidence
- compare the job against your CV
- generate a "why this company / why now" note
- optionally draft a 3-5 sentence recruiter message

That would turn this from a job alert into a real career-search agent.
