FROM nvcr.io/nvidia/pytorch:24.04-py3

# RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6  # OpenCV dependencies

WORKDIR /app

# Remove the opencv version shipped with the base image
# https://github.com/opencv/opencv-python/issues/884
# RUN pip uninstall -y opencv
# RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2/

COPY ./requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

### It will be more flexible to mount them on runtime
# COPY .env /app/.env
# COPY config /app/config
# COPY avr /app/avr

ENV PYTHONUNBUFFERED 1
ENTRYPOINT ["python"]
# CMD ["/bin/bash"]
