# Phishing Detector (Multimodal)

A Streamlit demo that classifies a single URL as phishing (malicious) or legitimate (safe) using a multimodal approach: a URL-based scikit-learn pipeline and an image-based Keras model running on a screenshot of the site.

This repository contains a small web UI (`app.py`) that:
- Accepts a website URL.
- Uses a pickled scikit-learn pipeline (`phishing.pkl`) to score URL-based features.
- Optionally fetches a screenshot of the URL and scores it using a MobileNetV2-based Keras model (`phishing_screenshot_mobilenetv2_focal.keras`).
- Fuses the two scores (weighted) and displays a final decision together with a confidence percentage and a progress bar.

---

## What changed / important UI details

- The app now shows a confidence percentage for the final fused score (displayed with one decimal place, e.g. `87.3% Confidence`).
- The app also displays a progress bar that reflects the final phishing score (0.0 to 1.0).
- Screenshot capture uses a Screenshot API (the app includes a placeholder `API_KEY` in `app.py` which you should replace with your own key). If screenshot retrieval fails the image model score falls back to a neutral value.

How the final score is computed in `app.py` (implementation detail):
- url_score = 1 - url_model_prob_for_safe_class  (the code uses the URL pipeline `predict_proba` result)
- img_score = image model output (0..1 where higher means more likely phishing)
- final_score = 0.6 * url_score + 0.4 * img_score
- final_score is shown as a percentage with one decimal and used to decide the label: > 0.5 -> "Phishing (Malicious)", else "Legitimate (Safe)".

---

## Project structure (important files)

- `app.py`  Streamlit application entrypoint (runs the UI).
- `phishing.pkl`  Pickled scikit-learn pipeline (preprocessing + URL model).
- `phishing_screenshot_mobilenetv2_focal.keras`  Keras image model used to score website screenshots.
- `requirements.txt`  Python dependencies for running the app.

---

## Quick start (Windows, cmd.exe)

1. (Optional) Create and activate a virtual environment:

   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```cmd
   pip install -r requirements.txt
   ```

3. Add your Screenshot API key (optional but recommended for image scoring):
   - Open `app.py` and replace the placeholder `API_KEY` with your real key for the screenshot service (or set an environment variable and modify the app to read it).

4. Run the Streamlit app:

   ```cmd
   streamlit run app.py
   ```

   The app will open in your browser (usually at http://localhost:8501).

---

## How to use the app

1. Paste a URL into the text box and click `Predict`.
2. The app will:
   - Show a short info message while it calls the URL and screenshot models.
   - Display the captured website screenshot (if retrieval succeeds).
   - Show the final decision label (Phishing / Legitimate) along with a confidence percentage and a progress bar representing the phishing probability.

Notes:
- If the screenshot service is unavailable, the image model score defaults to a neutral 0.5 so the URL model has more influence.
- The label semantics (which class corresponds to "phishing") depend on how `phishing.pkl` was trained; the app maps the pipeline probability into a phishing score as described above.

---

## Troubleshooting

- "FileNotFoundError: phishing.pkl": Ensure `phishing.pkl` is in the same folder where you run `streamlit run app.py`.
- Screenshot API errors: If you see a screenshot API error in the UI, verify your `API_KEY`, and the screenshot service availability.
- Streamlit not found: Ensure you installed the dependencies in the environment where you run the app.

---

## Security & privacy

The app sends the URL you enter to a screenshot service (remote) in order to capture a visual rendering of the page; the URL string is also passed to a local model. Do not paste credentials, personal tokens, or sensitive data into the URL input.

---

## Extending or re-training the models

- To update the URL model, retrain your scikit-learn pipeline and replace `phishing.pkl` with a new pickled pipeline.
- To update the image model, retrain and replace `phishing_screenshot_mobilenetv2_focal.keras`.

---

If you'd like, I can also modify `app.py` to read the screenshot API key from an environment variable (safer than hardcoding) and add that to the README; tell me if you want that.
