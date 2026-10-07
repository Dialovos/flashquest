"""vLLM 0.30 observation after first worker execution, possibly internal startup warmup."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from vllm.v1.worker.gpu_worker import Worker


class ValidationWorker(Worker):
    def execute_model(self, *args, **kwargs):
        output = super().execute_model(*args, **kwargs)
        path = os.environ.get("FLASHQUEST_VLLM_RUNTIME")
        if path and not getattr(self, "_validation_observed", False):
            torch.cuda.synchronize()
            model = self.model_runner.model
            tensors = [*model.parameters(), *model.buffers()]
            scales = []
            methods = set()
            for module in model.modules():
                method = getattr(module, "quant_method", None)
                if method is not None:
                    methods.add(type(method).__name__)
                if hasattr(module, "_k_scale") and hasattr(module, "_v_scale"):
                    scales.append({"k": float(module._k_scale.item()), "v": float(module._v_scale.item())})
            caches = [t for t in self.model_runner.kv_caches if isinstance(t, torch.Tensor)]
            record = {"method": "vllm0.30-after-first-worker-execution-v1",
                      "stage": "first-worker-execution; may be internal startup warmup",
                      "model_tensor_devices": sorted({str(t.device) for t in tensors}),
                      "model_parameter_devices": sorted({str(t.device) for t in model.parameters()}),
                      "quantization_methods": sorted(methods),
                      "cache_config_dtype": self.cache_config.cache_dtype,
                      "logical_cache_dtype": str(self.model_runner.kv_cache_dtype),
                      "cache_tensor_devices": sorted({str(t.device) for t in caches}),
                      "cache_storage_dtypes": sorted({str(t.dtype) for t in caches}),
                      "cache_tensor_bytes": sum(t.numel() * t.element_size() for t in caches),
                      "kv_scales": scales,
                      "cpu_offload_gb": self.vllm_config.offload_config.uva.cpu_offload_gb,
                      "offload_backend": self.vllm_config.offload_config.offload_backend,
                      "offload_group_size": self.vllm_config.offload_config.prefetch.offload_group_size}
            Path(path).write_text(json.dumps(record, allow_nan=False) + "\n")
            self._validation_observed = True
        return output
