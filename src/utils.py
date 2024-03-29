import numpy as np
import pandas as pd
from typing import Tuple


ylim_by_dataset = {
    "Student-Teacher": (1e-1, 1e8),
    "California Housing": (1e-2, 1e8),
    "Diabetes": (1e3, 3.16e7),
    "WHO Life Expectancy": (1e0, 1e11),
}


def generate_synthetic_data(
    return_X_y: bool,
    N: int = 1000,
    P: int = 30,
    D: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    X_bar = np.random.randn(N, P)
    X = X_bar[:, :D]
    beta_bar = np.random.randn(P, 1)
    Y = X_bar @ beta_bar
    return X, Y


def load_who_life_expectancy(**kwargs):
    # https://www.kaggle.com/kumarajarshi/life-expectancy-who

    life_expectancy_df = pd.read_csv("data/Life Expectancy Data.csv")
    life_expectancy_df.dropna(inplace=True)

    X = life_expectancy_df[
        [
            "Adult Mortality",
            "infant deaths",
            "Alcohol",
            "percentage expenditure",
            "Hepatitis B",
            "Measles ",
            " BMI ",
            "under-five deaths ",
            "Polio",
            "Total expenditure",
            "Diphtheria ",
            " HIV/AIDS",
            "GDP",
            "Population",
            " thinness  1-19 years",
            " thinness 5-9 years",
            "Schooling",
        ]
    ].values
    y = life_expectancy_df["Life expectancy "].values

    return X, y
