import os
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from dasa_fairseq.data.dep_bias_dataset import DepBiasDataset


@register_task("translation_dasa")
class TranslationTaskDASA(TranslationTask):
    """
    Same as TranslationTask, but wraps split datasets with DepBiasDataset.
    You must provide --dep-bias-dir containing train/valid/test folders:
      dep_bias/
        train/{idx}.npy
        valid/{idx}.npy
        test/{idx}.npy
    """

    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)
        parser.add_argument("--dep-bias-dir", type=str, default=None)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        super().load_dataset(split, epoch=epoch, combine=combine, **kwargs)
        if self.args.dep_bias_dir is None:
            raise ValueError("--dep-bias-dir is required for translation_dasa")
        dep_dir = os.path.join(self.args.dep_bias_dir, split)
        self.datasets[split] = DepBiasDataset(self.datasets[split], dep_dir)