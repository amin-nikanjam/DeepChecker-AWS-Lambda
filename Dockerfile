FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN yum -y install git

RUN pip3 install -r requirements.txt

COPY . ${LAMBDA_TASK_ROOT}

CMD ["handler.handler"]