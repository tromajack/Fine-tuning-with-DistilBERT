from sklearn.model_selection import train_test_split


def split_data(df, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    
    Args:
        data: Input DataFrame with 'text' and 'label' columns.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.
        
    Returns:
        X_train, X_test, y_train, y_test: Split datasets.
    """
    X = df.drop(['label'], axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test
