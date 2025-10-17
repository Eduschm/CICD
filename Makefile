install:
	pip install -r requirements.txt

train:
	python src/train.py

test:
	@echo "Running inference tests..."
	@echo "To be implemented: python src/test.py"

clean:
	@echo "Cleaning up generated artifacts..."
	rm -f models/.pkl
	rm -f results/.json

.PHONY: install train test clean