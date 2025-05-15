all:
	python scripts/orca_json_to_qm9_npz.py PFAS003.out -o data/pfas_qm9.npz

test: all
	pytest tests/test_qm9_npz_loader.py