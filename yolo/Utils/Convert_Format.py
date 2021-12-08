from os import path, makedirs
import pandas as pd
import numpy as np
import re
import os
from PIL import Image
from Get_File_Paths import GetFileList, ChangeToOtherMachine


def convert_vott_csv_to_yolo(
    vott_df,
    labeldict=dict(
        zip(
            ["zebra_line"],
            [
                0,
            ],
        )
    ),
    path="",
    target_name="data_train.txt",
    abs_path=False,
):
    print(labeldict)
    print(vott_df)
    # Encode labels according to labeldict if code's don't exist
    if not "code" in vott_df.columns:
        vott_df["code"] = vott_df["label"].apply(lambda x: labeldict[x])
    # Round float to ints
    for col in vott_df[["xmin", "ymin", "xmax", "ymax", "degree"]]:
        vott_df[col] = (vott_df[col]).apply(lambda x: round(x))

    # Create Yolo Text file
    last_image = ""
    txt_file = ""

    for index, row in vott_df.iterrows():
        print(row)
        if not last_image == row["filename"]:
            if abs_path:
                txt_file += "\n" + row["image_path"] + " "
            else:
                txt_file += "\n" + os.path.join(path, row["filename"]) + " "
            txt_file += ",".join(
                [
                    str(x)
                    for x in (row[["xmin", "ymin", "xmax", "ymax", "code", "degree"]].tolist())
                ]
            )
        else:
            txt_file += " "
            txt_file += ",".join(
                [
                    str(x)
                    for x in (row[["xmin", "ymin", "xmax", "ymax", "code", "degree"]].tolist())
                ]
            )
        last_image = row["filename"]
    file = open(target_name, "w")
    file.write(txt_file[1:])
    file.close()
    return True
