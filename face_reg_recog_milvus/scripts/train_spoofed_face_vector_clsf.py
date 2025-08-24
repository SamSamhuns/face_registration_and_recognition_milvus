"""
Train a DNN model to classify real vs fake/spoofed faces

train+test real+fake face images avai at https://github.com/SkyThonk/real-and-fake-face-detection
npy facenet vectors avai at https://drive.google.com/file/d/1JcVM81RvDycJWa-z9upXfiY7l9hSQ0Jf/view?usp=sharing

requirements:
    pip install scikit-learn
    pip install torch torchvision
        pip install pytorch-lightning

pytorch-lightning model componenents
    Initialization (__init__ and setup())
    forward step (tensor)
    Train Loop (training_step())
    Validation Loop (validation_step())
    Test Loop (test_step())
    Prediction Loop (predict_step())
    Optimizers and LR Schedulers (configure_optimizers())
"""

from collections import OrderedDict
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import accuracy
from torchvision import transforms as Transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights, resnet50

torch.set_float32_matmul_precision("high")


def unison_shuffled_copies(arr1, arr2):
    """
    Shuffle two numpy arrays in unison
    """
    assert len(arr1) == len(arr2)
    pidx = np.random.permutation(len(arr1))
    return arr1[pidx], arr2[pidx]


class FakeFaceImageModel(pl.LightningModule):
    """
    Module to train & classify if a face image is real or fake
    """

    def __init__(self, num_classes: int = 2, freeze_backbone: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        resnet_weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=resnet_weights)
        self.preprocess = resnet_weights.transforms()

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        # get final layer name
        final_layer = "classifier" if hasattr(self.model, "classifier") else "fc" if hasattr(self.model, "fc") else None
        if not final_layer:
            raise ValueError(f"Unrecognized final layer of feat ext {self.model.__name__}")

        n_inputs = getattr(self.model, final_layer).in_features
        classifier = nn.Sequential(
            OrderedDict([("dropout", nn.Dropout(p=0.2, inplace=False)), ("fc1", nn.Linear(n_inputs, num_classes))])
        )
        setattr(self.model, final_layer, classifier)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def _shared_eval_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        return loss, correct / total

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, sync_dist=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, sync_dist=True)
        return metrics


class FakeFaceVectorModel(pl.LightningModule):
    """
    Module to train & classify if a facenet 128 vector face-embedding is from a real or fake face
    Contains train, val, and pred functions
    """

    def __init__(self, n_feats: int = 128, **kwargs: Any) -> None:
        """
        Init model feature extractor backbone and final NN classifier
        """
        super().__init__(**kwargs)
        # self.classifier = nn.Linear(n_feats, n_classes)
        self.classifier = nn.Sequential(
            nn.Linear(n_feats, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 32),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = nn.BCELoss()(y_hat, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = nn.BCELoss()(y_hat, y)
        acc = accuracy(y_hat, y, task="binary", num_classes=2)
        return loss, acc

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, sync_dist=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, sync_dist=True)
        return metrics


def train_faces_with_cnn(train_dir: str, test_dir: str, max_epochs: int = 100, devices: int = 4):
    """
    Train faces from train and test directories with real & fake face images
    """
    # Define transformations to apply to the images
    transform = Transforms.Compose(
        [
            Transforms.ToTensor(),
        ]
    )

    # Load the image dataset using ImageFolder
    train_dataset = ImageFolder(train_dir, transform=transform)
    test_dataset = ImageFolder(test_dir, transform=transform)

    # Create data loaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=5, verbose=False, mode="max")

    model = FakeFaceImageModel()
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=devices,
        num_nodes=1,
        log_every_n_steps=5,
        callbacks=[early_stop_callback],
        check_val_every_n_epoch=5,
    )
    trainer.fit(model, train_loader, val_dataloaders=test_loader)
    trainer.test(model, dataloaders=test_loader)


def train_vectors_with_dnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_epochs: int = 150,
    devices: int = 4,
):
    """
    Use a DNN with pytorch lightining to classify face vectors
    """
    x_train, y_train = torch.Tensor(x_train), torch.Tensor(y_train)
    x_test, y_test = torch.Tensor(x_test), torch.Tensor(y_test)
    x_train, y_train = x_train.float(), y_train.float()
    x_test, y_test = x_test.float(), y_test.float()

    # load train & test dataset, dataloaders
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, num_workers=8, batch_size=64)

    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=5, verbose=False, mode="max")

    model = FakeFaceVectorModel()
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=devices,
        num_nodes=1,
        log_every_n_steps=5,
        callbacks=[early_stop_callback],
        check_val_every_n_epoch=10,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    trainer.test(model, dataloaders=test_loader)


def train_vectors_with_sklearn_clsf(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    """
    Train and eval svm, random forests & adaboost classifiers
    """

    def print_clf_train_test_acc(clf, X, y, acc_type):
        """
        Preds and prints the acc of a sklearn clsf
        """
        y_hat = clf.predict(X)
        acc = accuracy_score(y, y_hat)
        print(f"{acc_type} Accuracy: {acc:.4f}")
        scores = cross_val_score(clf, X, y, cv=5)
        print("\tCross val score avg: ", scores.mean())

    # SVM classifier
    svm_clf = svm.SVC()
    svm_clf.fit(x_train, y_train)
    print_clf_train_test_acc(svm_clf, x_train, y_train, "SVM Train")
    print_clf_train_test_acc(svm_clf, x_test, y_test, "SVM Test")

    # Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=0)
    rf_clf.fit(x_train, y_train)
    print_clf_train_test_acc(rf_clf, x_train, y_train, "RF Train")
    print_clf_train_test_acc(rf_clf, x_test, y_test, "RF Test")

    # Adaboost Classifier
    ab_clf = AdaBoostClassifier(n_estimators=150)
    ab_clf.fit(x_train, y_train)
    print_clf_train_test_acc(ab_clf, x_train, y_train, "Ada Train")
    print_clf_train_test_acc(ab_clf, x_test, y_test, "Ada Test")


if __name__ == "__main__":
    # train & test using face images
    TRAIN_DIR = "../downloads/real_fake_face_detection/train"
    TEST_DIR = "../downloads/real_fake_face_detection/test"
    train_faces_with_cnn(TRAIN_DIR, TEST_DIR)

    # train & test using face vectors extracted from a facenet model
    TRAIN_REAL_VEC = "../downloads/facenet_real_fake_face_vectors/train_real.npy"
    TRAIN_FAKE_VEC = "../downloads/facenet_real_fake_face_vectors/train_fake.npy"
    TEST_REAL_VEC = "../downloads/facenet_real_fake_face_vectors/test_real.npy"
    TEST_FAKE_VEC = "../downloads/facenet_real_fake_face_vectors/test_fake.npy"
    X_train_real = np.load(TRAIN_REAL_VEC)
    X_train_fake = np.load(TRAIN_FAKE_VEC)
    X_test_real = np.load(TEST_REAL_VEC)
    X_test_fake = np.load(TEST_FAKE_VEC)

    x_train_split = np.concatenate([X_train_real, X_train_fake])
    x_test_split = np.concatenate([X_test_real, X_test_fake])
    # zeros mean real, ones mean fake face images
    y_train_split = np.concatenate([np.zeros(len(X_train_real)), np.ones(len(X_train_fake))])
    y_test_split = np.concatenate([np.zeros(len(X_test_real)), np.ones(len(X_test_fake))])

    x_train_split, y_train_split = unison_shuffled_copies(x_train_split, y_train_split)
    x_test_split, y_test_split = unison_shuffled_copies(x_test_split, y_test_split)

    train_vectors_with_dnn(x_train_split, y_train_split, x_test_split, y_test_split)
    train_vectors_with_sklearn_clsf(x_train_split, y_train_split, x_test_split, y_test_split)
