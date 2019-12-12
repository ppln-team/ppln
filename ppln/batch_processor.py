class BaseBatchProcessor:
    def test_step(self, model, data, **kwargs):
        raise NotImplementedError

    def train_step(self, model, data, **kwargs):
        raise NotImplementedError

    def val_step(self, model, data, **kwargs):
        raise NotImplementedError
