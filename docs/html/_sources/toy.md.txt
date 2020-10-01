# Toy Mode

The `--full` switch specifies to run the NAS algorithm using full dataset and to
preconfigured epochs. This can sometimes take multiple days. For development purposes you can use toy mode.

```bash
python scripts/main.py
```

This will run all algorithms through very small batches and just for single epoch.