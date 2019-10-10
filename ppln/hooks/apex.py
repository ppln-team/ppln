from apex import amp

from ppln.hooks.optimizer import OptimizerHook


class ApexOptimizerHook(OptimizerHook):
    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()

        with amp.scale_loss(runner.outputs['loss'], runner.optimizer) as scaled_loss:
            scaled_loss.backward()

        if self.grad_clip is not None:
            self.clip_grads(amp.master_params(runner.optimizer))
        runner.optimizer.step()
