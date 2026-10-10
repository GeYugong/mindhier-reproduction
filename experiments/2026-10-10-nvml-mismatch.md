# GPU telemetry failure during Subject 1 formal training

At 2026-10-10 23:03 UTC, the status command failed because nvidia-smi returned exit 18: Driver/library version mismatch (NVML library 595.99). Raw process, memory and training logs were archived under runs/pipeline/gpu-status-failure-20261010T230311Z before any intervention.

The existing formal trainer and supervisor remained alive. Completed update records remained contiguous and finite, and each update retained 80 sample specifications. The latest checkpoint at step 1800 passed file-size and ZIP-directory checks; this does not constitute a fresh numerical restore verification. The exact observation is recorded in reports/nvml-mismatch-20261010/result.json.

No driver, shared environment, training process or training source was changed. GPU utilization and memory telemetry are unavailable while NVML is inconsistent. Continue monitoring training files and process liveness; host administration must coordinate any driver remediation with all running projects. Do not reboot or reload drivers during active training. The formal run remains bound to source 06e95b83a14b07fef51ebdb4c299b6df9914bb34, and a later recovery must preserve that source and wrapping.
