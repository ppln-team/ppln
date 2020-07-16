class BaseBatchProcessor:
    def test_step(self, model, batch, **kwargs):
        raise NotImplementedError

    def train_step(self, model, batch, **kwargs):
        raise NotImplementedError

    def val_step(self, model, batch, **kwargs):
        raise NotImplementedError
