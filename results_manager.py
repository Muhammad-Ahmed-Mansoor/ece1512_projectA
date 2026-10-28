from pathlib import Path
import json
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display


def save_and_show_results(project_dir, dataset, history, test_metrics, best_epoch, best_model_state):
    root = Path(project_dir) / 'results' / 'baseline'
    folder = root / dataset
    folder.mkdir(parents=True, exist_ok=True)

    (folder / 'metrics.json').write_text(json.dumps(
        {'best_epoch': best_epoch + 1, **{k: float(v) for k, v in test_metrics.items()}}, indent=2))
    df = pd.DataFrame(history)
    df['epoch'] += 1
    df.to_csv(folder / 'history.csv', index=False)
    torch.save(best_model_state, folder / 'best_model.pt')

    rows = []
    plt.figure(figsize=(7, 4))
    for name in ('C16', 'C17', 'BRACS'):
        path = root / name
        if not (path / 'metrics.json').exists():
            rows.append({'dataset': name, 'status': 'Not available'})
            continue
        metrics = json.loads((path / 'metrics.json').read_text())
        rows.append({'dataset': name, 'status': 'Available', **metrics})
        data = pd.read_csv(path / 'history.csv')
        line, = plt.plot(data['epoch'], data['train_loss'], '--', label=f'{name} train')
        plt.plot(data['epoch'], data['val_loss'], color=line.get_color(), label=f'{name} val')

    summary = pd.DataFrame(rows).set_index('dataset')
    display(summary)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return summary
