import threading
from itertools import chain

import torch
from torch.nn.parallel import DataParallel


class BalancedDataParallel(DataParallel):
    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    "module must have its parameters and buffers "
                    "on device {} (device_ids[0]) but found one of "
                    "them on device: {}".format(self.src_device_obj, t.device)
                )

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[: len(inputs)])
        return self.parallel_apply(replicas, inputs, kwargs)


class BalancedDataParallelCriterion(DataParallel):
    def forward(self, inputs, *targets, **kwargs):
        # Input should be already scattered, scattering the targets instead
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(inputs, *targets[0], **kwargs[0])
        inputs = [(input,) for input in inputs]
        replicas = self.replicate(self.module, self.device_ids[: len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, targets, kwargs)
        outputs = self.gather(outputs, self.output_device)
        return torch.mean(outputs)

    def parallel_apply(self, replicas, inputs, targets, kwargs):
        lock = threading.Lock()
        results = {}

        def _worker(i, module, input, target, kwargs, device=None):
            torch.set_grad_enabled(torch.is_grad_enabled())
            if device is None:
                device = input[0].get_device()
            try:
                with torch.cuda.device(device):
                    output = module(*input, *target, **kwargs)
                with lock:
                    results[i]: torch.Tensor = output
            except Exception as e:
                with lock:
                    results[i] = e

        if len(replicas) > 1:
            threads = [
                threading.Thread(target=_worker, args=(i, module, input, target, kwargs, device))
                for i, (module, input, target, kwargs, device) in enumerate(
                    zip(replicas, inputs, targets, kwargs, self.device_ids)
                )
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            _worker(0, replicas[0], tuple(inputs[0]), tuple(targets[0]), kwargs[0], self.device_ids[0])

        outputs = []
        for i in range(len(inputs)):
            output = results[i]
            if isinstance(output, Exception):
                raise output
            outputs.append(output)
        return outputs
