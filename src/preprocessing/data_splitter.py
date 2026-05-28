from sklearn.model_selection import train_test_split

from src.utils.config import (
    RANDOM_STATE,
    TEST_SIZE
)


class DataSplitter:
    """
    Handles train-test splitting for the machine learning pipeline.
    Uses stratified splitting to preserve class distribution.
    """

    def __init__(self, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y):
        """
        Split features and target into training and testing sets.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            X_train, X_test, y_train, y_test
        """

        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )