import pandas as pd

labels = pd.read_excel(
    r"C:\Users\User\PycharmProjects\JoNova\dataset\raw\KArSL\KArSL-502_Labels.xlsx"
)

sign_id = 160

result = labels[labels["SignID"] == sign_id]

print(result[["Sign-Arabic", "Sign-English"]])