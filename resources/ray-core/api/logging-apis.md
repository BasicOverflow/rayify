# Logging APIs

APIs for configuring logging in Ray jobs.

## LoggingConfig

Logging configuration for a Ray job. Applies to driver process and all worker processes.

```python
import ray

ray.init(
    logging_config=ray.LoggingConfig(
        encoding="TEXT",  # or "JSON"
        log_level="INFO",  # or "DEBUG"
        additional_log_standard_attrs=['name']
    )
)
```

**Parameters:**
- `encoding`: Encoding type ("TEXT" or "JSON")
- `log_level`: Log level ("DEBUG", "INFO", etc., default: "INFO")
- `additional_log_standard_attrs`: List of additional standard logger attributes to include

**Attributes:**
- `encoding`: Current encoding setting
- `log_level`: Current log level
- `additional_log_standard_attrs`: Additional attributes list

