$env:BACKEND_URL="http://localhost:8000"

Write-Host "Starting Backend..."
Start-Process -FilePath "..\.venv\Scripts\python.exe" -ArgumentList "-m uvicorn main:app --reload" -WorkingDirectory "backend"

Write-Host "Starting Frontend..."
Start-Process -FilePath "..\.venv\Scripts\python.exe" -ArgumentList "-m streamlit run app.py" -WorkingDirectory "frontend"

Write-Host "Both services have been started in separate windows."
