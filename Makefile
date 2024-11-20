.PHONY: install

poetry:
	pip install poetry
	poetry install

start:
	pre-commit install
	python hooks/run_bash.py
	poetry shell

test-all:
	@echo "Running all the unittests."
	python -m unittest tests.test_all.TestAll.test_run_all

test-sarima:
	@echo "Testing the SARIMA predictions for both datasets. This will take appr. 160 seconds."
	python -m unittest tests.test_sarima.TestAll.test_sarima

vm:
	@echo "Creating a virtual machine for the project."
	az network bastion rdp --enable-mfa true --name bas-irrekencapaciteit-net --resource-group rg-irrekencapaciteit-network --target-resource-id /subscriptions/903a8ea8-505d-4c45-a478-ab6649cb9e72/resourceGroups/rg-irrekencapaciteit-vms/providers/Microsoft.Compute/virtualMachines/vm-win1
