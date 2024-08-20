.PHONY: install

install:
	pre-commit install
	python hooks/run_bash.py
	poetry install
	poetry shell

test-eb:	# eb <- exhaustive both
	@echo "Testing both datasets exhaustive. This will take appr. 165 seconds."
	python tests/exhaustive/test_both.py

test-ec:	# ec <- exhaustive cumulative
	@echo "Testing the cumulative datasets exhaustive. This will take appr. 120 seconds."
	python tests/exhaustive/test_cumulative.py

test-ei:	# ei <- exhaustive individual
	@echo "Testing the individual datasets exhaustive. This will take appr. 80 seconds."
	python tests/exhaustive/test_individual.py

test-fb:	# fb <- fast both
	@echo "Testing both datasets fast. This will take appr. 100 seconds."
	python tests/fast/test_fast_both.py

test-fc:	# fc <- fast cumulative
	@echo "Testing the cumulative datasets fast. This will take appr. 70 seconds."
	python tests/fast/test_fast_cumulative.py

test-fi:	# fi <- fast individual
	@echo "Testing the individual datasets fast. This will take appr. 75 seconds."
	python tests/fast/test_fast_individual.py
