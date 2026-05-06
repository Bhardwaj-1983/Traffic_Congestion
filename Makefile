# Traffic Congestion — developer shortcuts
#
# Usage: make <target>
#
# Typical end-to-end workflow:
#     make install
#     make download              # optional: fetch 6 months of NYC data
#     make centroids             # optional: create zone centroid CSV
#     make pipeline              # run all 5 offline stages
#     make app                   # launch Streamlit dashboard
#
# Or with Docker:
#     make docker-build
#     make docker-run

PYTHON      ?= python
PIP         ?= pip
STREAMLIT   ?= streamlit
DOCKER      ?= docker
IMAGE_NAME  ?= traffic-congestion

# ── Help ────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo "Traffic Congestion — developer commands"
	@echo ""
	@echo "  install         Install Python requirements"
	@echo "  download        Download 6 months of NYC Yellow Taxi data"
	@echo "  centroids       Build taxi_zone_centroids.csv from TLC shapefile"
	@echo ""
	@echo "  pipeline        Run all 5 offline stages (load→preprocess→features→cluster→viz)"
	@echo "  load            Stage 1: data loader"
	@echo "  preprocess      Stage 2: preprocessing"
	@echo "  preprocess-iqr  Stage 2 with per-zone IQR speed filter (data-driven outliers)"
	@echo "  features        Stage 3: feature engineering"
	@echo "  cluster         Stage 4: clustering (K-Means, StandardScaler)"
	@echo "  cluster-robust  Stage 4 with RobustScaler (better for heavy-tailed features)"
	@echo "  cluster-dbscan  Stage 4 + DBSCAN hotspot detection"
	@echo "  viz             Stage 5: viz precomputation + static charts"
	@echo ""
	@echo "  app             Launch Streamlit dashboard"
	@echo ""
	@echo "  test            Run pytest suite"
	@echo "  lint            Quick compile-check on all src files"
	@echo "  clean           Remove build artefacts"
	@echo ""
	@echo "  docker-build    Build Docker image"
	@echo "  docker-run      Run dashboard via Docker"

# ── Environment ─────────────────────────────────────────────────────────────

.PHONY: install
install:
	$(PIP) install -r requirements.txt

# ── Data ingestion ──────────────────────────────────────────────────────────

.PHONY: download
download:
	$(PYTHON) scripts/download_data.py

.PHONY: centroids
centroids:
	$(PYTHON) scripts/make_centroids.py

# ── Pipeline stages ─────────────────────────────────────────────────────────

.PHONY: load preprocess features cluster viz
load:
	$(PYTHON) -m src.data_loader

preprocess:
	$(PYTHON) -m src.preprocessing

preprocess-iqr:
	$(PYTHON) -m src.preprocessing --iqr

features:
	$(PYTHON) -m src.feature_engineering

cluster:
	$(PYTHON) -m src.clustering

cluster-robust:
	$(PYTHON) -m src.clustering --scaler robust

cluster-dbscan:
	$(PYTHON) -m src.clustering --dbscan

viz:
	$(PYTHON) -m src.visualization

.PHONY: pipeline
pipeline:
	$(PYTHON) main.py

# ── Streamlit app ───────────────────────────────────────────────────────────

.PHONY: app
app:
	$(STREAMLIT) run app/app.py --server.port 8502

# ── Quality gates ───────────────────────────────────────────────────────────

.PHONY: test
test:
	$(PYTHON) -m pytest tests/ -v

.PHONY: lint
lint:
	$(PYTHON) -m py_compile $(shell find src app scripts -name '*.py')
	@echo "All modules compile cleanly."

.PHONY: clean
clean:
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

# ── Docker ──────────────────────────────────────────────────────────────────

.PHONY: docker-build
docker-build:
	$(DOCKER) build -t $(IMAGE_NAME) .

.PHONY: docker-run
docker-run:
	$(DOCKER) run --rm -p 8502:8502 \
		-v $(CURDIR)/data:/app/data \
		-v $(CURDIR)/models:/app/models \
		-v $(CURDIR)/outputs:/app/outputs \
		$(IMAGE_NAME)
