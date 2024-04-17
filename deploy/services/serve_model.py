import json
import os
import subprocess
import sys

import ray
import requests
from ray import serve

sys.path.append(".")

from madewithml.config import MODEL_REGISTRY  # NOQA: E402
from madewithml.serve import ModelDeployment  # NOQA: E402

# # Copy from S3
github_username = os.environ.get("GITHUB_USERNAME")
subprocess.check_output(["aws", "s3", "cp", f"s3://madewml/{github_username}/mlflow/", str(MODEL_REGISTRY), "--recursive"])
subprocess.check_output(["aws", "s3", "cp", f"s3://madewml/{github_username}/results/", "./", "--recursive"])

# Entrypoint
run_id = [line.strip() for line in open("run_id.txt")][0]
entrypoint = ModelDeployment.bind(run_id=run_id, threshold=0.9)

ray.init(runtime_env={"env_vars": {"GITHUB_USERNAME": os.environ["GITHUB_USERNAME"]}}, num_gpus=1)
serve.run(entrypoint)


# Query
title = "Transfer learning with transformers"
description = "Using transformers for transfer learning on text classification tasks."
json_data = json.dumps({"title": title, "description": description})
res = requests.post("http://127.0.0.1:8000/predict/", data=json_data).json()
print(res)
