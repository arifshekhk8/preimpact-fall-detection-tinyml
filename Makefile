PY := ./.venv/bin/python

.PHONY: sync probe preprocess e1 e2 e3 e5 export all tables lint

sync:        ; $(PY) scripts/sync_fdlib.py
probe:       ; $(PY) scripts/run_kernel.py nb00_probe
preprocess:  ; $(PY) scripts/run_kernel.py nb01_preprocess
e1:          ; $(PY) scripts/run_kernel.py nb02_e1
e2:          ; $(PY) scripts/run_kernel.py nb03_e2
e3:          ; $(PY) scripts/run_kernel.py nb04_e3
e5:          ; $(PY) scripts/run_kernel.py nb05_e5
export:      ; $(PY) scripts/run_kernel.py nb06_export

# One kernel at a time, halting at the first failure (rules K2/K3).
all: sync
	$(PY) scripts/run_kernel.py nb01_preprocess nb02_e1 nb03_e2 nb04_e3 nb05_e5 nb06_export

tables:      ; $(PY) scripts/build_tables.py
lint:        ; $(PY) -m py_compile src/fdlib/*.py src/fdlib/datasets/*.py kaggle/*/*.py
