# MediaGuard

MediaGuard is a Streamlit app that analyzes uploaded images using Microsoft Azure AI and classifies them as `safe`, `sensitive`, `unsafe`, or `analysis_failed`.

Requirements
- Python 3.10+

Environment variables (required)
- `AZURE_AI_ENDPOINT` — full Azure AI endpoint URL to POST images to
- `AZURE_AI_KEY` — API key
- `AZURE_AI_REGION` — Azure region

Install
```
pip install -r requirements.txt
```

Run
```
streamlit run app.py --server.port=8000 --server.address=0.0.0.0
```

Notes
- The app reads credentials only from environment variables (or a loaded .env). Do not hardcode secrets.
- If Azure AI fails or returns an unexpected JSON schema, the app will return `analysis_failed` and will not produce guessed classifications.
