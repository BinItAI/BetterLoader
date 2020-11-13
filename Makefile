install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

sdist:
	python setup.py sdist

upload:
	twine upload dist/*

sample:
	python examples/example.py

clean:
	rm -rf *.egg-info
	rm -rf dist/

test:
	python tests/tests.py

deploy: clean sdist upload
