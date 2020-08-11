install:
	python3 -m pip install --upgrade pip
	pip3 install -r requirements.txt

run_sample:
	python3 examples/example.py

run_test:
	python3 tests/tests.py

sdist:
	python setup.py sdist

upload:
	twine upload dist/*

sample: install run_sample

test: install run_test

deploy: sdist upload