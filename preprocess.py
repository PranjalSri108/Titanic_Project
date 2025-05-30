import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path, dtype=str)

def clean_data(df):
    df = df.drop_duplicates()

    df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')

    df = df.drop(columns=['Cabin'])

    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Pclass'] = pd.to_numeric(df['Pclass'], errors='coerce')
    
    age_group = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(age_group)
    df['Age'] = df['Age'].fillna(df['Age'].median())

    if df['Embarked'].mode().shape[0] > 1:
        df['Embarked'] = df['Embarked'].fillna('S')
    else:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    return df

def engineer_features(df):
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
    df['Title'] = df['Title'].replace({
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Lady': 'Rare', 'Countess': 'Rare',
        'Capt': 'Rare', 'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare',
        'Rev': 'Rare', 'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare'
    })
    title_counts = df['Title'].value_counts()
    rare_titles = title_counts[title_counts < 10].index
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')

    df['FamilySize'] = df['SibSp'].astype(int) + df['Parch'].astype(int) + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    df['AgeBin'] = pd.qcut(df['Age'], q=5, labels=[f'AgeBin{i}' for i in range(1, 6)], duplicates='drop')
    df['FareBin'] = pd.qcut(df['Fare'], q=5, labels=[f'FareBin{i}' for i in range(1, 6)], duplicates='drop')

    return df

def encode_and_scale(df):
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Pclass', 'Title', 'AgeBin', 'FareBin'], drop_first=False)

    df = df.drop(columns=['Name', 'Ticket', 'PassengerId'])

    fare_cap = df['Fare'].quantile(0.99)
    age_cap = df['Age'].quantile(0.99)
    df['Fare'] = np.minimum(df['Fare'], fare_cap)
    df['Age'] = np.minimum(df['Age'], age_cap)

    df['Fare'] = (df['Fare'] - df['Fare'].min()) / (df['Fare'].max() - df['Fare'].min())
    df['Age'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())

    return df

def save_outputs(df, output_dir='~/heethrark_internship/ml_projects/titanic_project/output'):
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(f'{output_dir}/cleaned.csv', index=False)

    X = df.drop(columns=['Survived'], errors='ignore').values
    np.save(f'{output_dir}/final_features.npy', X)

