import warnings

from ppln.hooks.optimizer import OptimizerHook

try:
    from apex import amp
except ImportError as e:
    warnings.warn(
        f"Error \"{e}\" during importing apex library. To use mixed precison"
        ' you should install it from https://github.com/NVIDIA/apex'
    )


class ApexOptimizerHook(OptimizerHook):
    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()

        with amp.scale_loss(runner.outputs['loss'], runner.optimizer) as scaled_loss:
            scaled_loss.backward()

        if self.grad_clip is not None:
            self.clip_grads(amp.master_params(runner.optimizer))
        runner.optimizer.step()
