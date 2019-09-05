init:
	pip install -r requirements.txt

test:
	nosetests3 -s tests
