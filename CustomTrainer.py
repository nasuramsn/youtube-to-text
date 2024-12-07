from transformers import Trainer


class CustomTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir

        # self.model = self.model.to('cpu')  # モデルをCPUに移動

        # すべてのテンソルを連続化する
        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

        super().save_model(output_dir, _internal_call)
