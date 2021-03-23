import pandas as pd

df = pd.read_csv("output_emo/result0.csv")

print(df[df["frame_number"] == 3]["prediction"].values)
print(df["frame_number"][1] - df["frame_number"][0])