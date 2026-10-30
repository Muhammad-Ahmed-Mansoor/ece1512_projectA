from pathlib import Path
import json
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from IPython.display import display


def save_and_show_results(project_dir, dataset, history, test_metrics, best_epoch, best_model_state,
                          model, test_loader, device):
    root = Path(project_dir) / 'results' / 'baseline'
    folder = root / dataset
    folder.mkdir(parents=True, exist_ok=True)

    (folder / 'metrics.json').write_text(json.dumps(
        {'best_epoch': best_epoch + 1, **{k: float(v) for k, v in test_metrics.items()}}, indent=2))
    df = pd.DataFrame(history)
    df['epoch'] += 1
    df.to_csv(folder / 'history.csv', index=False)
    torch.save(best_model_state, folder / 'best_model.pt')

    # Save predictions for the current dataset (model already holds the best weights).
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch['input'].to(device, dtype=torch.float32)).argmax(1).cpu().tolist()
            predictions.extend(zip(batch['slide_id'], batch['label'].tolist(), pred))
    pd.DataFrame(predictions, columns=['slide_id', 'true', 'pred']).to_csv(
        folder / 'predictions.csv', index=False)

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
    plt.title('Training and Validation Losses for All Datasets')
    plt.legend()
    plt.show()

    # Show confusion matrices for datasets with saved predictions.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.suptitle('Confusion Matrices')
    for ax, name in zip(axes, ('C16', 'C17', 'BRACS')):
        file = root / name / 'predictions.csv'
        if file.exists():
            data = pd.read_csv(file)
            ConfusionMatrixDisplay.from_predictions(
                data['true'], data['pred'], labels=range({'C16': 2, 'C17': 4, 'BRACS': 7}[name]),
                ax=ax, colorbar=False)
        else:
            ax.text(0.5, 0.5, 'Not available', ha='center', va='center')
            ax.axis('off')
        ax.set_title(name)
    plt.tight_layout()
    plt.show()
    return summary
