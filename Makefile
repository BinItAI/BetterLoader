install:
	python3 -m pip install --upgrade pip
	pip3 install -r requirements.txt

run_sample:
	python3 examples/example.py

run_test:
	python3 tests/tests.py

sample:
	install run_sample

test:
	install run_test

sdist:
	python setup.py sdist

upload:
	twine upload dist/*

deploy:
	sdist upload