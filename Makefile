install:
	pip install -r requirements.txt

run:
	FLASK_APP=app.py flask run --host=0.0.0.0 --port=3000