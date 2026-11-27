# Iris MLOps demo: FastAPI, Streamlit, MLflow, Docker

An Iris `RandomForestClassifier` is trained and served through a FastAPI API. Training runs are tracked with MLflow, the model is registered as `iris-classifier`, and a Streamlit page calls the API.

## Project files

```text
main.py                 FastAPI API: GET / and POST /predict
train.py                training, MLflow logging, model registration
app.py                  Streamlit frontend calling the API
model.joblib            latest trained model used by the API
requirements.txt        pinned direct dependencies
Dockerfile              API image
Dockerfile.frontend     Streamlit image
docker-compose.yml      API + frontend services
screenshots/            screenshots for submission
```

## 1. Set up Python on Windows

The checked setup uses Python 3.14.7. MLflow 3.18.0 and Streamlit 1.67.0 installed successfully in the Python 3.14.7 environment used for this project.

```powershell
py -3.14 -m venv .venv-mlops
.\.venv-mlops\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Each dependency listed in `requirements.txt` is pinned to a version observed in `pip freeze` from the working environment. (A complete Windows freeze also contains Windows-only dependencies, so the file lists the app's direct dependencies instead.)

## 2. Train, track and register three models

Run these from the project root. Each invocation creates an MLflow run in experiment `iris-classification`, logs its parameters, accuracy, weighted F1, the MLflow model and `model.joblib`, then creates another `iris-classifier` registry version.

```powershell
python .\train.py --n-estimators 50 --max-depth 3 --random-state 42
python .\train.py --n-estimators 100 --max-depth 5 --random-state 42
python .\train.py --n-estimators 150 --max-depth none --random-state 21
```

`--max-depth none` means no depth limit. The SQLite tracking and registry database is `mlflow.db` in the project root. Training also refreshes `model.joblib`; the API loads this file when it starts.

Start the MLflow UI in a second PowerShell window:

```powershell
.\.venv-mlops\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open <http://localhost:5000>. Inspect the `iris-classification` experiment for the three runs, parameters, metrics and artifacts; open the Model Registry and inspect the three versions of `iris-classifier`.

## 3. Run and check the API locally

Start the API in another PowerShell window:

```powershell
.\.venv-mlops\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000/docs> to try the endpoints. Example request:

```powershell
$body = @{ sepal_length = 5.1; sepal_width = 3.5; petal_length = 1.4; petal_width = 0.2 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType 'application/json' -Body $body
```

The response contains an integer `prediction` and a string `class_name`.

## 4. Run the Streamlit page locally

In another PowerShell window, set `API_URL` to the local API (this is the default) and start Streamlit:

```powershell
$env:API_URL = 'http://localhost:8000'
.\.venv-mlops\Scripts\streamlit.exe run .\app.py
```

Open <http://localhost:8501>, enter four measurements, and click **Predict**. The page displays the predicted class or a clear connection/timeout/request error.

## 5. Run both services with Docker Compose

Stop the local API first so host port 8000 is free. Build and start the API and frontend:

```powershell
docker compose up --build
```

Open <http://localhost:8000/docs> for FastAPI and <http://localhost:8501> for Streamlit. To stop the services, press `Ctrl+C`, then run:

```powershell
docker compose down
```

Inside Compose, `api` is the API service's DNS name on the shared Compose network, so the frontend uses `http://api:8000`. `localhost` inside the frontend container would point back to the frontend container itself. From the Windows host browser, use `localhost` because Compose publishes both ports to the host.

## Python 3.12/3.13 fallback

Python 3.14.7 worked for this project's installed packages. If a future package or machine cannot install them under 3.14, install Python 3.13 (or 3.12), recreate the environment, and match both Docker base images to that version:

```powershell
deactivate
py -3.13 -m venv .venv-mlops313
.\.venv-mlops313\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip freeze
```

Update the first line of both `Dockerfile` and `Dockerfile.frontend` to `FROM python:3.13-slim`, then rebuild using `docker compose up --build`. Re-pin dependencies from the new environment's real installed versions if they differ.

## What to commit

Commit source files, Docker configuration, `requirements.txt`, `README.md`, screenshots, and the small `model.joblib` needed to start the API. Do not commit `venv/`, `.venv-mlops/`, `mlruns/`, or `mlflow.db`: these are local environments/tracking state and are excluded by `.gitignore`. The SQLite DB contains the experiment and registry history, so take screenshots of its UI before cleanup. If you need to transfer run history, use an explicit export/backup rather than adding generated tracking state to the assignment repository.

## Screenshots for submission

Save screenshots under `screenshots/`:

1. `api-docs.png` — FastAPI Swagger page at `/docs` with a successful `/predict` response.
2. `streamlit-prediction.png` — Streamlit inputs and the displayed prediction.
3. `mlflow-runs.png` — MLflow experiment listing three runs with parameters and accuracy/weighted F1 metrics.
4. `mlflow-run-artifacts.png` — one run's artifacts showing the MLflow model and `model.joblib`.
5. `mlflow-registry.png` — registered `iris-classifier` and its three versions.
6. `docker-compose.png` — running Compose services or terminal showing both services started, plus the pages at ports 8000 and 8501 if the submission asks for UI proof.

## Submission checklist

- [ ] Three training runs appear in the MLflow UI with distinct parameters.
- [ ] Each run logs accuracy, weighted F1, MLflow model and `model.joblib`.
- [ ] Registry contains `iris-classifier` versions 1, 2 and 3.
- [ ] `/predict` responds and Streamlit displays that response.
- [ ] `docker compose up --build` starts both services and both pages load.
- [ ] Add the screenshots and repository link requested by the course platform.

## Be ready to explain

- **Why use weighted F1 as well as accuracy?** Accuracy is the fraction of all correct predictions; weighted F1 combines precision and recall for each class and weights classes by their support.
- **Why does each training create a new version?** Logging the model with `registered_model_name="iris-classifier"` asks MLflow to register that run's model; each successful registration gets the next version number.
- **Why is the Compose API URL `http://api:8000`?** Compose gives each service a DNS name on its shared network; `api` resolves to the API container, while `localhost` in the frontend container points to itself.

## Technologies

Python, scikit-learn, joblib, FastAPI, Uvicorn, Streamlit, Requests, MLflow, SQLite, Docker Compose.
