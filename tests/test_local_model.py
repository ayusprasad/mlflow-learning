from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def test_local_wine_model_trains_without_tracking_service() -> None:
    dataset = load_wine()
    features_train, features_test, target_train, target_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=42, stratify=dataset.target
    )
    model = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)
    model.fit(features_train, target_train)
    assert accuracy_score(target_test, model.predict(features_test)) >= 0.8
