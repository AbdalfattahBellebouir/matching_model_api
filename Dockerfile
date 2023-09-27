FROM python:3

ENV apiport 7777
WORKDIR /matching_model_api
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
CMD [ "chmod 0755 start_api.sh" ]
ENTRYPOINT [ "./start_api.sh"]
