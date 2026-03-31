import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from timeit import default_timer as timer
from tqdm.auto import tqdm

from src import CRNN, MJSynthCustom, global_variables, label_utils

device = "cuda" if torch.cuda.is_available() else "cpu"


def edit_distance(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def character_accuracy(pred_seqs, true_seqs):
    batch_char_acc = 0.0

    for pred, true in zip(pred_seqs, true_seqs):
        distance = edit_distance(pred, true)
        sample_char_acc = max(0.0, 1 - distance / max(1, len(true)))
        batch_char_acc += sample_char_acc

    return batch_char_acc / len(true_seqs)


def train_step(model, dataloader, loss_fn, optimizer):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_char_acc = 0.0

    for X, y, y_lengths in dataloader:
        X = X.to(device)
        y = y.to(device)
        y_lengths = y_lengths.to(device)

        y_pred = model(X)

        input_lengths = torch.full(
            size=(X.size(0),),
            fill_value=y_pred.size(0),
            dtype=torch.long,
            device=device,
        )

        loss = loss_fn(y_pred, y, input_lengths, y_lengths)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_seqs = label_utils.decode_prediction(y_pred.detach().cpu())
        true_seqs = [
            label_utils.target_to_list(target.cpu(), length.cpu())
            for target, length in zip(y, y_lengths)
        ]
        batch_acc = sum(pred == true for pred, true in zip(pred_seqs, true_seqs)) / len(true_seqs)
        batch_char_acc = character_accuracy(pred_seqs, true_seqs)
        train_acc += batch_acc
        train_char_acc += batch_char_acc

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    train_char_acc /= len(dataloader)
    return train_loss, train_acc, train_char_acc



def test_step(model, dataloader, loss_fn):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_char_acc = 0.0

    with torch.inference_mode():
        for X, y, y_lengths in dataloader:
            X = X.to(device)
            y = y.to(device)
            y_lengths = y_lengths.to(device)

            test_pred = model(X)

            input_lengths = torch.full(
                size=(X.size(0),),
                fill_value=test_pred.size(0),
                dtype=torch.long,
                device=device,
            )

            loss = loss_fn(test_pred, y, input_lengths, y_lengths)
            test_loss += loss.item()

            pred_seqs = label_utils.decode_prediction(test_pred.cpu())
            true_seqs = [
                label_utils.target_to_list(target.cpu(), length.cpu())
                for target, length in zip(y, y_lengths)
            ]
            batch_acc = sum(pred == true for pred, true in zip(pred_seqs, true_seqs)) / len(true_seqs)
            batch_char_acc = character_accuracy(pred_seqs, true_seqs)
            test_acc += batch_acc
            test_char_acc += batch_char_acc

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    test_char_acc /= len(dataloader)
    return test_loss, test_acc, test_char_acc



def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, model_save_path, epochs=5):
    results = {
        "train_loss": [],
        "train_acc": [],
        "train_char_acc": [],
        "test_loss": [],
        "test_acc": [],
        "test_char_acc": [],
    }
    best_test_acc = None

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_char_acc = train_step(model, train_dataloader, loss_fn, optimizer)
        test_loss, test_acc, test_char_acc = test_step(model, test_dataloader, loss_fn)

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"train_char_acc: {train_char_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            f"test_char_acc: {test_char_acc:.4f}"
        )

        results["train_loss"].append(float(train_loss))
        results["train_acc"].append(float(train_acc))
        results["train_char_acc"].append(float(train_char_acc))
        results["test_loss"].append(float(test_loss))
        results["test_acc"].append(float(test_acc))
        results["test_char_acc"].append(float(test_char_acc))

        if best_test_acc is None or test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()

    if args.epochs < 1:
        args.epochs = 5

    if args.batch_size < 1:
        args.batch_size = 32

    samples = max(1, args.samples)
    test_samples = (samples + 7) // 8

    # Assumes root points to the 90kDICT32px folder
    root = Path(args.root)
    train_annotation = root / "annotation_train.txt"
    test_annotation = root / "annotation_test.txt"

    train_dataset = MJSynthCustom.MJSynthCustom(
        root,
        train_annotation,
        samples,
    )

    test_dataset = MJSynthCustom.MJSynthCustom(
        root,
        test_annotation,
        test_samples,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    crnn_model = CRNN.CRNN(
        input_channels=1,
        output_shape=global_variables.MAX_LABEL_LEN * (len(global_variables.CHAR_LIST) + 1),
    ).to(device)

    loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(crnn_model.parameters(), lr=0.001)

    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_index = 0
    model_save_path = models_dir / f"CRNN_{model_index}.pth"
    while model_save_path.exists():
        model_index += 1
        model_save_path = models_dir / f"CRNN_{model_index}.pth"

    start_time = timer()

    training_results = train(
        model=crnn_model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        model_save_path=model_save_path,
        epochs=args.epochs,
    )

    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")
    print(f"Best model saved to {model_save_path}")
