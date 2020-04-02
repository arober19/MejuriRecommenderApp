FROM python:3.6
ADD . /deploy
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./feature_vectorslastone.csv /deploy/
COPY ./recommendermodel /deploy/
COPY ./templates /deploy/
#COPY ./templates/index.html /deploy/
#COPY ./templates/results.html /deploy/
WORKDIR /deploy/
RUN pip install --trusted-host pypi.python.org -r requirements.txt
#EXPOSE 5000
#CMD ["gunicorn", "-b", '0.0.0.0:8000', "app"]
EXPOSE 80
#CMD ["python", "app.py", "-m", "flask", "run", "--host=0.0.0.0"]
ENTRYPOINT ["python", "app.py", "-m", "flask", "run", "--host=0.0.0.0"]