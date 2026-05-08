# smart-document-analyser

This repo is set up to deploy on Render as two services:

- `smart-document-analyser-api` for the FastAPI backend
- `smart-document-analyser-web` for the Streamlit frontend

The large local model in `backend/model/` is not required by the current app. The backend uses OpenRouter, so that folder should stay out of GitHub.

## Required Render environment variables

Backend service:

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL` optional, defaults to `openai/gpt-4o-mini`
- `APP_BASE_URL` optional, set this to your frontend Render URL
- `APP_TITLE` optional
- `FRONTEND_ORIGIN` set this to your frontend Render URL

Frontend service:

- `BACKEND_URL` set this to your backend Render URL

## Local run

Backend:

```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

Frontend:

```powershell
cd frontend
$env:BACKEND_URL="http://localhost:8000"
..\.venv\Scripts\python.exe -m streamlit run app.py
```

## Push without the large model

If `backend/model/` was previously committed, remove it from Git tracking but keep it on disk:

```powershell
git rm -r --cached backend/model
git add .gitignore README.md render.yaml backend frontend requirements.txt
git commit --amend --no-edit
git push --force-with-lease origin main
```
