# ---------- 前端构建 ----------
# 之前需要先在宿主机 cd web && npm run build，现在直接在镜像里构建，
# docker build . 即可自洽完成。
FROM node:20-slim AS web-builder

WORKDIR /web

COPY ./web/package.json ./web/package-lock.json ./
RUN npm install

COPY ./web ./
RUN npm run build

# ---------- 运行时 ----------
# 原来的 python:3.10.13-slim-bullseye 的 security 源已经 404，换 bookworm。
FROM python:3.10-slim-bookworm

WORKDIR /app

# copy dist of web
COPY --from=web-builder /web/dist /app/web/dist
COPY ./free_one_api /app/free_one_api
COPY ./requirements.txt ./main.py /app/

# install git
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/

# g4f 8.1.6 的 provider 需要较新的 curl_cffi,否则会报 BaseSession() unexpected keyword 'proxy'
# (re_gpt 会把它钉在 0.5.9,这里强制升上去,re_gpt 那条链路本来也不在个人网关里用)
RUN pip install --no-cache -r requirements.txt \
    && pip uninstall torch tensorflow transformers triton -y \
    && rm -rf /usr/local/lib/python3.10/site-packages/nvidia \
    && pip install --no-cache-dir --no-deps -U "curl_cffi>=0.7"

CMD [ "python", "main.py" ]
