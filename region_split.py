import pandas as pd
PATH_TO_EICU_DATA = "/data/datasets/physionet.org/files/eicu-crd/2.0/"
PATH_TO_MAPPER = "/data/datasets/new_split/root/episode_mapper.csv"
ROOT = "/data/datasets/new_split/"#path to new_splits/
SUFFIX = "2"
IGNORE_UNDER = 15
print("Part 3: Regional Splits")
suffix=SUFFIX

def split(name, write=False):
    len_tr = 0
    len_val = 0
    len_test = 0
    train_list = pd.read_csv(f"{name}/train_listfile{suffix}.csv")
    test_list = pd.read_csv(f"{name}/test_listfile{suffix}.csv")
    val_list = pd.read_csv(f"{name}/val_listfile{suffix}.csv")
    hospitals = pd.read_csv(PATH_TO_EICU_DATA+"hospital.csv.gz")
    pats = pd.read_csv(PATH_TO_EICU_DATA+"patient.csv.gz")
    mapper_df = pd.read_csv(PATH_TO_MAPPER)
    mapper = dict(zip(mapper_df['episode'], mapper_df['unitstayid']))

    comp_table = pd.concat([train_list, test_list, val_list])
    comp_table["patientunitstayid"] = comp_table["stay"].apply(
        lambda x: mapper[x] #lambda x: int(x.split("_")[0])
    )

    tr = train_list["stay"].apply(
        lambda x:  mapper[x] #int(x.split("_")[0])
        ).tolist()
    test = test_list["stay"].apply(
        lambda x:  mapper[x] #int(x.split("_")[0])
        ).tolist()
    val = val_list["stay"].apply(
        lambda x:  mapper[x] #int(x.split("_")[0])
        ).tolist()
    print("Train: ", len(tr))
    print("Val: ", len(val))
    print("Test: ", len(test))
    print("\n")
    print("Total: ", len(tr) + len(val) + len(test))
    print("------------------")

    tr_pats = pats[pats["patientunitstayid"].isin(tr)]
    test_pats = pats[pats["patientunitstayid"].isin(test)]
    val_pats = pats[pats["patientunitstayid"].isin(val)]

    merged_tr = pd.merge(tr_pats, hospitals, on="hospitalid")
    merged_test = pd.merge(test_pats, hospitals, on="hospitalid")
    merged_val = pd.merge(val_pats, hospitals, on="hospitalid")

    print("Train:")
    size = merged_tr.groupby("region").size()
    print(size)
    print("\n")

    print("Validation:")
    size = merged_val.groupby("region").size()
    print(size)
    print("\n")

    print("Test:")
    size = merged_test.groupby("region").size()
    print(size)

    if write == True:

        #############################################################
        # Create and save train listfiles
        patients_visited = set()
        for region, region_group in merged_tr.groupby("region"):
            region = region.lower()
            table = comp_table[
                comp_table["patientunitstayid"].isin(region_group["patientunitstayid"])
            ]
            region_table = table[
                [col for col in table.columns if col != "patientunitstayid"]
            ]


            #ENSURE NO PATIENT IS IN ANY OTHER REGION -- CONRAD
            pid = region_table['stay'].apply(lambda x: x.split("_")[0])
            if pid.isin(patients_visited).astype(int).sum() > 0: 
                print("TRIGGERED", pid[pid.isin(patients_visited)])
                region_table = region_table[~pid.isin(patients_visited)]
            patients_visited = patients_visited | set(pid)


            region_table.to_csv(f"{name}_split/{region}_train{suffix}.csv", index=False)
            len_tr += len(region_table)

        #############################################################
        # Create and save validation listfiles
        patients_visited = set()
        for region, region_group in merged_val.groupby("region"):
            region = region.lower()
            table = comp_table[
                comp_table["patientunitstayid"].isin(region_group["patientunitstayid"])
            ]
            region_table = table[
                [col for col in table.columns if col != "patientunitstayid"]
            ]

            #ENSURE NO PATIENT IS IN ANY OTHER REGION -- CONRAD
            pid = region_table['stay'].apply(lambda x: x.split("_")[0])
            if pid.isin(patients_visited).astype(int).sum() > 0: 
                print("TRIGGERED", pid[pid.isin(patients_visited)])
                region_table = region_table[~pid.isin(patients_visited)]
            patients_visited = patients_visited | set(pid)

            region_table.to_csv(f"{name}_split/{region}_val{suffix}.csv", index=False)
            len_val += len(region_table)

        #############################################################
        # Create and save test listfiles
        patients_visited = set()
        for region, region_group in merged_test.groupby("region"):
            region = region.lower()
            table = comp_table[
                comp_table["patientunitstayid"].isin(region_group["patientunitstayid"])
            ]
            region_table = table[
                [col for col in table.columns if col != "patientunitstayid"]
            ]


            #ENSURE NO PATIENT IS IN ANY OTHER REGION -- CONRAD
            pid = region_table['stay'].apply(lambda x: x.split("_")[0])
            if pid.isin(patients_visited).astype(int).sum() > 0: 
                print("TRIGGERED", pid[pid.isin(patients_visited)])
                region_table = region_table[~pid.isin(patients_visited)]
            patients_visited = patients_visited | set(pid)


            region_table.to_csv(f"{name}_split/{region}_test{suffix}.csv", index=False)
            len_test += len(region_table)

    print("\n")
    print("Training samples:")
    print(len_tr)
    print("Validation samples:")
    print(len_val)
    print("Validation samples:")
    print(len_val)
    print("Test samples:")
    print(len_test)

for task in ['phenotyping', 'in-hospital-mortality', 'decompensation','length-of-stay']:
    print(task)
    split(ROOT+task, True)
    print()
    print()