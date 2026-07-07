FROM python
WORKDIR /app
COPY . /app

RUN pip install awscli

RUN pip install -r requirements.txt
CMD ["python", "application.py"]
