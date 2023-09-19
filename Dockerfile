FROM python:3

WORKDIR /app


COPY . /app
 
RUN pip3 install --no-cache-dir -r requirements.txt


CMD [ "python", "./mic.py" ]

ENV Token_VALUE="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"

