import pandas as pd

def get_invalid_pairs():
    pr_df = pd.read_csv("./data/product_data.csv")
    at_df = pd.read_csv("./data/attribute_data.csv")

    attrs = at_df["attribute_name"].unique()
    types = pr_df["des_product_type"].unique()

    print(attrs)
    print(types)

    tdict = {}
    adict={}

    count = 0
    for t in pr_df["cod_modelo_color"]:
        tdict[t] = pr_df.loc[count, "des_product_type"]
        count += 1

    count = 0

    for a in attrs:
        adict[a] = set()
    for a in at_df["attribute_name"]:
        adict[a].add(tdict[at_df.loc[count, "cod_modelo_color"]])
        count += 1

    pairs = []

    for a in attrs:
        for t in types:
            if not (t in adict[a]):
                pairs.append([t, a])

    return pairs

p = get_invalid_pairs()
print(p)



