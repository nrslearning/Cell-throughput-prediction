# pull python base image
FROM python:3.10-slim

# copy application files
ADD /cell_TP_pred_api /cell_TP_pred_api/

# specify working directory
WORKDIR /cell_TP_pred_api

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app/main.py"]