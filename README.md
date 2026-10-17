# RAG Profile Matching

Match job descriptions to resumes with section-level embeddings. Resumes are loaded through the Milestone 1 file tools in `fs_tools.py`. Chunks are embedded with the Hugging Face model `all-MiniLM-L6-v2` and stored in a local ChromaDB collection.

## Layout

| File | Role |
|---|---|
| `fs_tools.py` | `read_file` and `list_files` from Milestone 1 |
| `dataset.py` | 43 candidates and 6 job descriptions |
| `generate_resumes.py` | Writes `resumes/`, `jobs/`, and `eval_labels.json` |
| `resume_rag.py` | Section chunking, metadata, embeddings, ChromaDB |
| `job_matcher.py` | Hybrid search, must-have filter, 0-100 score, JSON |
| `rag_experiment.ipynb` | Precision, recall, must-have violations, latency |
| `eval_labels.json` | Resumes that meet each job before any search is run |

## Setup

Python 3.9 or newer.

```powershell
cd C:\Users\nagar\Downloads\rag-profile-matching
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python generate_resumes.py
python resume_rag.py
```

If pip fails with an SSL certificate error:

```powershell
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

The first index build downloads `all-MiniLM-L6-v2`. On this network the HTTPS certificate check fails, so `resume_rag.py` downloads the model without verifying that certificate, the same workaround the file assistant uses for Groq.

## Match a job

```powershell
python job_matcher.py jobs/senior_python_backend.txt
```

Add `--semantic` to rank by embedding similarity only, with no skill or years filter:

```powershell
python job_matcher.py jobs/senior_python_backend.txt --semantic
```

Hybrid scoring is 60% best-section similarity, 25% share of job skills found, and 15% years versus the minimum. The result is clamped to 0-100. MiniLM cosine for a real section match usually falls between 0.45 and 0.80, so that band is stretched across the similarity portion of the score. Anyone missing a must-have skill, or below the minimum years, is left out of `top_matches`.

`jobs/senior_python_backend.txt` asks for Python, Django, and 5+ years. Nagaraj has Django and Python with 4 years, so hybrid search leaves him out. Semantic search can still rank him highly because the wording is close.

## Notebook

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name rag-profile-matching --display-name "Python (rag-profile-matching)"
jupyter notebook rag_experiment.ipynb
```

The notebook rebuilds the index and compares semantic and hybrid search on all six jobs.

- Precision@10: labeled matches among the resumes returned. If the filter returns fewer than 10, the denominator is the number returned.
- Recall@10: labeled matches that appear in the results, divided by all labeled matches.
- Must-have violations: returned resumes that miss a required skill or the years bar.
- Latency: median milliseconds to embed the job and rank resumes, over 3 runs. Index build time is separate.

Labels in `eval_labels.json` are the people who already meet that job's skills, years, and role. They were written with the dataset, not taken from the matcher output.

## Demo (3-4 minutes)

1. Open one resume and show the section chunks from the notebook.
2. Run `python job_matcher.py jobs/senior_python_backend.txt` and show the JSON.
3. Point out that Nagaraj (4 years, Django) is absent, while the 5+ year Django resumes remain.
4. Show the notebook table: hybrid violations at 0, and semantic precision below hybrid precision.

## Submit

Push this folder to GitHub. Do not commit `.venv/` or `chroma_db/`. Record the demo video separately and submit the video plus the repository link.
