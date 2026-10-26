FROM public.ecr.aws/lambda/python:3.12

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pidetect/ ${LAMBDA_TASK_ROOT}/pidetect/
COPY model/ ${LAMBDA_TASK_ROOT}/model/

ENV MODEL_DIR=${LAMBDA_TASK_ROOT}/model \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

CMD ["pidetect.lambda_handler.handler"]
