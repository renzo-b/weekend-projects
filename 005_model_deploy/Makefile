.PHONY: app
app:
	uvicorn app.api:app --host 0.0.0.0 --port 5000 --reload --reload-dir app

.PHONY: install
install:
	python -m pip install -e . --no-cache-dir