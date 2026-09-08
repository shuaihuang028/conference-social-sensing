PYTHON := .venv/bin/python
PYTHON_BOOTSTRAP ?= python3.12
PYTHONPATH := src
DATA_CONFIG ?= configs/data.json

.PHONY: setup dataset baselines train figures test serve all

setup:
	$(PYTHON_BOOTSTRAP) -m venv .venv
	.venv/bin/pip install -r requirements.txt

dataset:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m conference_social_sensing.data.build_dataset --config $(DATA_CONFIG)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m conference_social_sensing.data.split_dataset --config configs/split.json
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m conference_social_sensing.data.build_group_crops --config configs/group_crops.json

baselines:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m conference_social_sensing.baselines.run_baselines --config configs/baselines.json

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m conference_social_sensing.modeling.train_group_model --config configs/group_model.json

figures:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/generate_report_figures.py

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

serve:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m conference_social_sensing.api.serve --checkpoint artifacts/group_model/best_model.pt

all: dataset baselines train figures test
