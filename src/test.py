import pickle
for v in ["v5", "v6"]:
    with open(f"models/explorer/{v}/vecnormalize.pkl", "rb") as f:
        vn = pickle.load(f)
    print(v, "std du retour normalisé :", vn.ret_rms.var ** 0.5)