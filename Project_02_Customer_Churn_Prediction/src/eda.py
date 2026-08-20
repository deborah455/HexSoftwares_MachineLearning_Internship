import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_customer_churn.csv"
SCREENSHOT_DIR = BASE_DIR / "screenshots"

SCREENSHOT_DIR.mkdir(exist_ok=True)


df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# Basic information
# --------------------------------------------------

print("=" * 60)
print("CUSTOMER CHURN EDA")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum().sum())


# --------------------------------------------------
# Churn distribution
# --------------------------------------------------

print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nChurn percentage:")
print(
    (df["Churn"].value_counts(normalize=True) * 100).round(2)
)


# --------------------------------------------------
# Churn by contract
# --------------------------------------------------

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn rate by contract:")
print(contract_churn.round(2))


plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "eda_contract_churn.png",
    dpi=150
)

plt.show()


# --------------------------------------------------
# Churn by Internet Service
# --------------------------------------------------

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn rate by internet service:")
print(internet_churn.round(2))


plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "eda_internet_churn.png",
    dpi=150
)

plt.show()


# --------------------------------------------------
# Churn by tenure
# --------------------------------------------------

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Tenure Distribution by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Tenure (Months)")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "eda_tenure_churn.png",
    dpi=150
)

plt.show()


# --------------------------------------------------
# Monthly charges
# --------------------------------------------------

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Monthly Charges")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "eda_monthly_charges.png",
    dpi=150
)

plt.show()


# --------------------------------------------------
# Churn by payment method
# --------------------------------------------------

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn rate by payment method:")
print(payment_churn.round(2))


plt.figure(figsize=(11, 6))

sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn"
)

plt.title("Customer Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "eda_payment_churn.png",
    dpi=150
)

plt.show()


# --------------------------------------------------
# Numerical correlation
# --------------------------------------------------

numeric_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]

correlation = df[numeric_columns].corr()

print("\nNumerical correlation:")
print(correlation.round(3))


plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Numerical Feature Correlation")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "eda_correlation.png",
    dpi=150
)

plt.show()


print("\nEDA complete.")
print(f"Charts saved to: {SCREENSHOT_DIR}")