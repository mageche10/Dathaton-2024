import pandas as pd

# For each attribute, which values are possibilities?
attribute_data = pd.read_csv('./data/attribute_data.csv')

possible = attribute_data.groupby("attribute_name")["des_value"].unique()

print(possible.reset_index())

print("count: " + str(len(attribute_data["attribute_name"].unique())))

# Relations between product category and families/type

product_data = pd.read_csv('./data/product_data.csv')

relations = product_data.groupby("des_product_category")["des_product_type"].unique()

print(relations.reset_index())



