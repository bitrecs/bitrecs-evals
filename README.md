# bitrecs-evals

Docker Image:

**ghcr.io/janusdotai/bitrecs-evals:main**

```
docker pull ghcr.io/bitrecs/bitrecs-evals:main
```

## docker run

```


docker run --rm --env-file $(pwd)/.env -v $(pwd)/output:/app/output --workdir /app ghcr.io/bitrecs/bitrecs-evals:main sh -c "python bitrecs_eval_runner.py"

```

## docker compose

```
docker compose build
docker compose up
```

## python direct

```
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 bitrecs_eval_runner.py

```