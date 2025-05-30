import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import os
from pathlib import Path
import csv
import json

# Import your models dynamically based on the configuration
def get_model(model_name, input_size, num_classes):
    if model_name == "VerySimpleNN":
        from models.simple_nn import VerySimpleNN
        return VerySimpleNN(input_size, num_classes)
    elif model_name == "NNWithHidden":
        from models.nn_with_hidden import NNWithHidden
        return NNWithHidden(input_size, num_classes)
    elif model_name == "NNWithDropout":
        from models.nn_with_dropout import NNWithDropout
        return NNWithDropout(input_size, num_classes)
    # Add other models here
    else:
        raise ValueError(f"Unknown model: {model_name}")

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    output_dir = os.getcwd()  # this will now be a unique timestamped directory
    logs_dir = Path(output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    file_path = Path(get_original_cwd()) / "3_classes_processed_data.csv"
    df = pd.read_csv(file_path)
    label_encoders = {}
    for column in ['ductility', 'roof', 'relative_position']:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

    # Encode the target variable
    label_encoder_target = LabelEncoder()
    df['structural_system'] = label_encoder_target.fit_transform(df['structural_system'])

    # Define features and target
    X = df.drop(columns=['structural_system'])
    y = df['structural_system']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    input_size = X_train_scaled.shape[1]
    num_classes = len(label_encoder_target.classes_)

    # Initialize the model
    model = get_model(cfg.model_config.model_name, input_size, num_classes)
    criterion = nn.CrossEntropyLoss()

    # Initialize the optimizer
    if cfg.model_config.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=cfg.model_config.lr, momentum=cfg.model_config.momentum)
    elif cfg.model_config.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.model_config.lr)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.model_config.optimizer}")

    # Get the output directory from Hydra
    output_dir = os.getcwd()
    logs_dir = Path(output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Save the configuration to the logs directory
    config_path = logs_dir / "config.yaml"
    with open(config_path, "w") as config_file:
        OmegaConf.save(cfg, config_file)

    # Open a CSV file to save the loss values
    loss_file_path = logs_dir / "loss_values.csv"
    with open(loss_file_path, mode='w', newline='') as loss_file:
        loss_writer = csv.writer(loss_file)
        loss_writer.writerow(["Epoch", "Loss"])

        # Training loop
        for epoch in range(cfg.model_config.num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            # Write the loss value to the CSV file
            loss_writer.writerow([epoch + 1, loss.item()])
            print(f"Epoch {epoch+1}/{cfg.model_config.num_epochs}, Loss: {loss.item()}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)

    y_pred_decoded = label_encoder_target.inverse_transform(predicted.numpy())
    y_test_decoded = label_encoder_target.inverse_transform(y_test_tensor.numpy())

    accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
    report = classification_report(y_test_decoded, y_pred_decoded)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)

    report_path = logs_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Save structured JSON version (for parsing/analysis)
    report_dict = classification_report(y_test_decoded, y_pred_decoded, output_dict=True)
    json_report_path = logs_dir / "classification_report.json"
    with open(json_report_path, "w") as json_file:
        json.dump(report_dict, json_file, indent=4)

if __name__ == "__main__":
    main()
