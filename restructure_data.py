import pandas as pd
import os

product_data = pd.read_csv('./data/product_data.csv')

relations = product_data["des_product_category"].unique()
print(relations)

for cat in relations:
    if not os.path.exists("./data/images/" + str(cat)):
        os.makedirs("./data/images/" + str(cat))