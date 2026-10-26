
import zipfile
import json
import os
import tempfile

source_model = "Artifacts/BiGRU_model.keras"
fixed_model = "Artifacts/BiGRU_model_fixed.keras"

with zipfile.ZipFile(source_model, "r") as source:
    config = json.loads(source.read("config.json"))

    # Remove the unsupported field from layer configurations.
    def clean_config(obj):
        if isinstance(obj, dict):
            obj.pop("quantization_config", None)
            for value in obj.values():
                clean_config(value)
        elif isinstance(obj, list):
            for item in obj:
                clean_config(item)

    clean_config(config)
    fixed_config = json.dumps(config).encode("utf-8")

    with zipfile.ZipFile(fixed_model, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)

            if item.filename == "config.json":
                data = fixed_config

            target.writestr(item, data)

print("Created compatibility-fixed model:", fixed_model)
