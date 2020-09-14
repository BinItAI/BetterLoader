install:
	python3 -m pip install --upgrade pip
	pip3 install -r requirements.txt

docs:
	python setup.py sdist

upload:
	twine upload dist/*

sample:
	python3 examples/example.py

clean:
	rm -rf *.egg-info
	rm -rf dist/

test:
	python3 tests/tests.py

lint:
	pylint ./betterloader/

deploy: clean docs upload