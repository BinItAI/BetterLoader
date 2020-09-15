install:
	python3 -m pip install --upgrade pip
	pip3 install -r requirements.txt

deploy-docs:
	cd docs && GIT_USER=$(GIT_USER) USE_SSH=true yarn deploy

upload:
	twine upload dist/*

sample:
	python3 examples/example.py

clean:
	rm -rf *.egg-info
	rm -rf dist/

test:
	python3 tests/tests.py

deploy: clean deploy-docs upload