def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score, roc_auc_score
    import os
    import sys

    # Setup path
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(ROOT_DIR)

    from models.siamese_nn import SiameseMLP
    from my_datasets.siamese_separated_dataset import SiamesePairCSV

    # Paths
    train_csv = os.path.join(ROOT_DIR, "data/speaker_pairs/train_pairs.csv")
    val_csv = os.path.join(ROOT_DIR, "data/speaker_pairs/val_pairs.csv")

    # Datasets & Loaders
    train_dataset = SiamesePairCSV(train_csv)
    val_dataset = SiamesePairCSV(val_csv)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Model, loss, optimizer
    model = SiameseMLP()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    EPOCHS = 20
    model.train()
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_preds, train_labels = 0.0, [], []

        for x1, x2, label in train_loader:
            pred = model(x1, x2)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x1.size(0)
            train_preds.extend(pred.detach().squeeze().tolist())
            train_labels.extend(label.squeeze().tolist())

        val_loss, val_preds, val_labels = 0.0, [], []
        model.eval()
        with torch.no_grad():
            for x1, x2, label in val_loader:
                pred = model(x1, x2)
                loss = criterion(pred, label)
                val_loss += loss.item() * x1.size(0)
                val_preds.extend(pred.squeeze().tolist())
                val_labels.extend(label.squeeze().tolist())
        model.train()

        # Metrics
        train_acc = accuracy_score(train_labels, [1 if p > 0.5 else 0 for p in train_preds])
        val_acc = accuracy_score(val_labels, [1 if p > 0.5 else 0 for p in val_preds])
        try:
            train_auc = roc_auc_score(train_labels, train_preds)
            val_auc = roc_auc_score(val_labels, val_preds)
        except ValueError:
            train_auc = val_auc = None

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f" Train Loss: {train_loss/len(train_dataset):.4f} | Acc: {train_acc*100:.2f}% | AUC: {train_auc:.4f}" if train_auc else "  ⚠️ Train AUC not available")
        print(f" Val   Loss: {val_loss/len(val_dataset):.4f} | Acc: {val_acc*100:.2f}% | AUC: {val_auc:.4f}" if val_auc else "  ⚠️ Val AUC not available")

    # Save the final model
    model_dir = os.path.join(ROOT_DIR, "saved_siamese")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "siamese_mlp_final.pt")
    torch.save(model.state_dict(), model_path)
    print(f" Model saved to {model_path}")




# if __name__ == "__main__":
#     train_siamese_model()