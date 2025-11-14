import os
import time
from torch.utils.tensorboard import SummaryWriter

# Always write under repo_root/runs
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
run_dir = os.path.join(REPO_ROOT, 'runs', f"smoke_test_{time.strftime('%Y%m%d-%H%M%S')}")
os.makedirs(run_dir, exist_ok=True)

writer = SummaryWriter(log_dir=run_dir)
print(f'Writing TensorBoard events to: {run_dir}')

# Write a few simple scalars and a text so TensorBoard has something to load
for step in range(5):
    writer.add_scalar('smoke/value', step * step, step)
writer.add_text('smoke/info', 'TensorBoard smoke test: scalars + text written', 0)
writer.flush()
writer.close()

print('Done. Now run:\n  tensorboard --logdir runs\nthen open the printed URL (e.g., http://localhost:6006).')