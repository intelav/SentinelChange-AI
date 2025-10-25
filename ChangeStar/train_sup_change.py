import ever as er
import numpy as np
import torch
from tqdm import tqdm

er.registry.register_all()


# The registration callback now points to our renamed function
def register_evaluate_fn(launcher):
    # RENAMED: Point to the more generic function name
    launcher.override_evaluate(evaluate_change_detection)


# RENAMED: from evaluate_levircd to a more generic name
def evaluate_change_detection(self, test_dataloader, config=None):
    self.model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # This metric is generic for any binary pixel-level task
    metric_op = er.metric.PixelMetric(2,
                                      self.model_dir,
                                      logger=self.logger)

    with torch.no_grad():
        for img, ret_gt in tqdm(test_dataloader):
            img = img.to(device)

            # This model forward pass is generic
            change = self.model.module(img).sigmoid() > 0.5

            pr_change = change.cpu().numpy().astype(np.uint8)
            # The key 'change' matches the output of our new SYSUCDLoader
            gt_change = ret_gt['change']
            gt_change = gt_change.numpy()
            y_true = gt_change.ravel()
            y_pred = pr_change.ravel()

            y_true = np.where(y_true > 0, np.ones_like(y_true), np.zeros_like(y_true))

            metric_op.forward(y_true, y_pred)

    metric_op.summary_all()
    torch.cuda.empty_cache()


# The rest of the file (including the commented-out xview2 parts) can remain the same
try:
    from core import field
except ImportError:
    print("Warning: Could not import 'field' from core.field. Assuming default mask key.")
    class MockField:
        MASK1 = 'mask'
    field = MockField()



if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    trainer = er.trainer.get_trainer('th_amp_ddp')()
    # This line correctly calls register_evaluate_fn, which now sets up our renamed function
    blob = trainer.run(after_construct_launcher_callbacks=[register_evaluate_fn])