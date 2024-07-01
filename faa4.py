PATH_TO_PATIENT = "/data/datasets/physionet.org/files/eicu-crd/2.0/patient.csv.gz"
ROOT = "/data/datasets/new_split/"#path to new_splits/
EICU_OUTPUT = "./output/" #output directory from data_extraction_root
PATH_TO_EPISODE_MAPPER ="/data/datasets/new_split/root/episode_mapper.csv"
SUFFIX = "2"
IGNORE_UNDER = 15
#################
import pandas as pd
import numpy as np
import os
import sys 

DIRECTORIES = ["root/"]
p = 1
proc_total = 1
if len(sys.argv) > 1:
    p, proc_total = sys.argv[1].split('/')
    p, proc_total = int(p), int(proc_total)
    
if not ("issues" in set(os.listdir("./")) or "issues/" in set(os.listdir("./"))):
    os.mkdir("./issues")

if not ("old" in set(os.listdir("./issues")) or "old/" in set(os.listdir("./issues"))):
    os.mkdir("./issues/old/")

for filename in [f"inconsistent{p}.txt", f"listfile2_err{p}.txt", f"audit_err{p}.txt", f"truncate{p}.txt"]:
    with open(f"./issues/{filename}", "w") as f:
        f.write("")

CONFIRMED_ALREADY = set()
# for file in os.listdir("./issues"):
#     if "visited" in file:
#         with open("./issues/"+file, "r") as f:
#             for line in f:
#                 CONFIRMED_ALREADY.add(line)

#######################
pats = pd.read_csv(PATH_TO_PATIENT)
pats['age'] = pats['age'].replace({'> 89':90}).fillna(-1).astype(int)
#should be order of features in mimic timeseries files

#younger than 18 and more than one icu stay in hospital admit is excluded
e = pats[pats['age'] >= 18]
cohort_count = e.groupby(by='patienthealthsystemstayid').count()
index_cohort = cohort_count[cohort_count['patientunitstayid'] == 1].index
e = e[e['patienthealthsystemstayid'].isin(index_cohort)]

eligible_patients = set(e['patientunitstayid'])

pats = pats.set_index('patientunitstayid')
pats['uniquepid'] = pats['uniquepid'].apply(lambda x: x.split("-")[0]+x.split("-")[1])

listfiles = {}

all_patients = {"phenotyping_split/":set(), "in-hospital-mortality_split/":set() ,"length-of-stay_split/":set(), "decompensation_split/":set()}

for task in ["phenotyping_split/", "in-hospital-mortality_split/" ,"length-of-stay_split/", "decompensation_split/"]:
    seen_tr = set()
    seen_ts = set()
    seen_val = set()
    all_df = []
    for REGION in [ "south", "midwest", "west", "northeast"]:
        lf_tr = pd.read_csv(f"{ROOT}{task}/{REGION}_train{SUFFIX}.csv")
        lf_ts = pd.read_csv(f"{ROOT}{task}/{REGION}_test{SUFFIX}.csv")
        lf_val = pd.read_csv(f"{ROOT}{task}/{REGION}_val{SUFFIX}.csv")

        uq_tr = set(lf_tr['stay'].apply(lambda x: x.split("_")[0])) 
        uq_ts = set(lf_ts['stay'].apply(lambda x: x.split("_")[0])) 
        uq_val = set(lf_val['stay'].apply(lambda x: x.split("_")[0])) 

        all_df.append(lf_tr)
        all_df.append(lf_ts)
        all_df.append(lf_val)

        for uq_sub in [uq_ts, uq_tr, uq_val]:
            for uq_broad in [seen_tr, seen_ts, seen_val]:
                assert len(uq_sub & uq_broad) == 0, task+REGION+" data leakage"

        assert len(uq_ts & uq_val) == 0, task+REGION+" data leakage"
        assert len(uq_tr & uq_ts) == 0, task+REGION+" data leakage"
        assert len(uq_tr & uq_val) == 0, task+REGION+" data leakage"
        
        
        seen_tr |= uq_tr
        seen_ts |= uq_ts
        seen_val |= uq_val

        all_patients[task] |= set(lf_tr['stay'])
        all_patients[task] |= set(lf_ts['stay'])
        all_patients[task] |= set(lf_val['stay'])
    
    listfiles[task] = pd.concat(all_df)

    assert len(listfiles[task]) != 0, task+" listfile is empty"


######################
ct=0
import pandas as pd
import os
import numpy as np

#should be order of features in mimic timeseries files
ORDER = [
            "itemoffset",
            "Capillary Refill",
            "Invasive BP Diastolic",
            "FiO2",
            "Eyes",
            "Motor",
            "GCS Total",
            "Verbal",
            "glucose",
            "Heart Rate",
            "admissionheight",
            "MAP (mmHg)",
            "O2 Saturation",
            "Respiratory Rate",
            "Invasive BP Systolic",
            "Temperature (C)",
            "admissionweight",
            "pH",
        ]

mapper_df = pd.read_csv(PATH_TO_EPISODE_MAPPER)
#maps filename to unitstayid
mapper = dict(zip(mapper_df['episode'], mapper_df['unitstayid']))

assert len(mapper_df['unitstayid']) == len(set(mapper_df['unitstayid'])), "mapper points to same unitid"
assert len(mapper_df['episode']) == len(set(mapper_df['episode'])), "mapper points to same episode"

def audit(df, file):
    if len(df) == 0:
        return
    #[TEST] Ensure columns are in proper order
    assert list(df.columns) == ORDER, file+ " columns out of order"

    #[TEST] Ensure no missing timestamps
    assert df.isnull()['itemoffset'].sum() == 0, file + " has null itemoffsets"

    #[TEST] Ensure file has non-empty rows
    assert not np.any(df[df.columns[1:]].isnull().all(axis=1)), file + " has empty rows"

    #[TEST] Ensure timestamps are non-negative
    assert df['itemoffset'].min() >= 0, file + " has negative itemoffsets"

    #[TEST] Ensure timestamps are distinct
    if len(df) > 1:
        assert np.min(np.array(df['itemoffset'][1:]) - np.array(df['itemoffset'][:-1])) > 1/61, file + " has timestamps too close together"

    #[TEST] Ensure timestamps are sorted
    assert list(df['itemoffset']) == sorted(list(df['itemoffset'])), file + " has timestamps out of order"
    
    #[TEST] Ensure categorical variables are correct
    assert np.all(df['Eyes'].dropna().isin([1,2,3,4,np.nan])), f"{file} eyes not in range"
    assert np.all(df['Verbal'].dropna().isin([1,2,3,4,5,np.nan])), f"{file} verbal not in range"
    assert np.all(df['GCS Total'].dropna().isin(list(range(3,16))+[np.nan])), f"{file} total not in range"
    assert np.all(df['Motor'].dropna().isin([1,2,3,4,5,6,np.nan])), f"{file} motor not in range"
    assert np.all(df['Capillary Refill'].dropna().isin([1,0,np.nan])), f"{file} CR not in range"

    #[TEST] Height and weight only recorded once
    assert df['admissionheight'].count() < 2, f"{file} height >= 2 instances"
    assert df['admissionweight'].count() < 2, f"{file} weight >= 2 instances"
    
    #[TEST] Ensure recordings are at offset 0
    if df['admissionheight'].count() == 1:
        assert not np.isnan(df['admissionheight'][0]), f"{file} first height entry is non-numeric"
        assert float(df['itemoffset'][0]) == 0, f"{file} first height entry is at itemoffset 0"
    if df['admissionweight'].count() == 1:
        assert not np.isnan(df['admissionweight'][0]), f"{file} first eeight entry is non-numeric"
        assert float(df['itemoffset'][0]) == 0, f"{file} first weight entry is at itemoffset 0"

def meet_ihm_criteria(df, los, expire, unitid, file):
    #[TEST] non-empty and eligble patient and has 1+ record < 48 hrs and LOS >= 48 hrs and expired is not null
    assert len(df) > 0 and unitid in eligible_patients and (df['itemoffset'].min() <= 48) and los >= 48 and expire < 2, file+" basic exclusion not met"
    ihm = listfiles['in-hospital-mortality_split/']
    ihm = ihm[ihm['stay'] == file]

    #[TEST] ensures match with what listfile says and no duplicates in list files
    same_record = (expire == ihm['y_true'][0])
    assert same_record, file+" expire binary mismatch"
    assert len(ihm) == 1, file+" more than one entry in listfile"

LOS_GB = listfiles['length-of-stay_split/'].groupby('stay')
LOS_GB_min = LOS_GB.min()
LOS_GB_max = LOS_GB.max()
def meet_los_criteria(df, los, expire, unitid, file):
    #[TEST] non-empty and eligible patient and LoS > 5
    assert len(df) > 0 and unitid in eligible_patients and los >= 5, file+" basic exclusion not met"

    #[TEST] Atleast 1 record below the minimum period_length in listfile
    assert df['itemoffset'].min() < float(LOS_GB_min['period_length'].loc[file]), file+" basic exclusion not met"

    assert float(LOS_GB_min['period_length'].loc[file]) >= 5, file +" must have sample >= 5 hrs"

    #[TEST] Ensures max period_length is near los (since max period_length is int and LoS is float)
    mx = float(LOS_GB_max['period_length'].loc[file])
    assert los < mx + 1 and mx - 1 < los, file+" LoS in listfile is incorrect"

def meet_pheno_criteria(df, los, expire, unitid, file):
    assert len(df) > 0 and df['itemoffset'].min() <= los and unitid in eligible_patients, file+" basic exclusion not met"
    pheno_lf = listfiles['phenotyping_split/'].set_index('stay')
    assert int(pheno_lf.loc[file]['period_length']) == int(los), file+" period length off"

DECOMP_GB = listfiles['decompensation_split/'].groupby('stay')
DECOMP_GB_min = DECOMP_GB.min()
DECOMP_GB_max = DECOMP_GB.max()

def meet_decomp_criteria(df, los, expire, unitid, file):
    #[TEST] non-empty and eligible patient and LoS > 5
    assert len(df) > 0 and unitid in eligible_patients and los >= 5, file+" basic exclusion not met"
    
    #[TEST] Atleast 1 record below the minimum period_length in listfile
    assert df['itemoffset'].min() < DECOMP_GB_min['period_length'].loc[file], file+" basic exclusion not met"

    #[TEST] Ensures max period_length is near los (since max period_length is int and LoS is float)
    mx = float(DECOMP_GB_max['period_length'].loc[file])
    assert los < mx + 1 and mx - 1 < los, file+" LoS in listfile is incorrect"

    decomp_lf = listfiles['decompensation_split/']
    d = decomp_lf[decomp_lf['stay'] == file]
    d = d[d['period_length'] >= los - 24]
    d_n = decomp_lf[decomp_lf['stay'] == file]
    d_n = d_n[d_n['period_length'] < los - 24]

    #[Test] last 24 hours match expire status
    assert np.all(d['y_true'] == expire), file+" last 24 hours do not match expire status"
    if len(d_n) > 0:
        assert np.all(d_n['y_true'] == 0), file+"before last 24 hours != 0"


visited = set()
ct=0
for task in DIRECTORIES:

    for dir in ['']:#['train/', 'test/']:

        ITER = os.listdir(ROOT+task+dir)
        num_files = len(ITER)
        num_files //= proc_total
        ITER = sorted(ITER)[(p-1)*num_files:(p)*num_files+1]
        
        n = len(ITER)
        for i,file in enumerate(ITER):
            print(task,dir,round(i/n*100,4), end="%\r")
            
            if ".csv" not in file or file in visited or "listfile" in file or "mapper" in file:
                continue

            visited.add(file)
            unitid = mapper[file]

            #[TEST] Validate mapper
            pat_row = pats.loc[unitid]
            tmp = file.split("_")[0]

            los = float(pat_row['unitdischargeoffset'])
            expire_status = 0; d = {'Expired':1, "Alive":0}
            if str(pat_row['unitdischargestatus']) not in d:
                expire_status = 2
            else:
                expire_status = d[str(pat_row['unitdischargestatus'])]
            

            df2 = pd.read_csv(ROOT+task+dir+file)

            if len(df2) < IGNORE_UNDER:
                for key in all_patients:
                    if file in all_patients[key]:
                        print(f"ERROR (see issues/listfile2_err{p}):", file, "not supposed to be in", key)
                        with open(f"./issues/listfile2_err{p}.txt", "a") as f:
                            f.write(f"{file} has length {len(df2)} and is in {key}\n")
                         

                continue

            before = len(df2)
            los/=60
            los+=1e-4
            df2 = df2[df2['itemoffset'] <= los]
            after = len(df2)

            tocsv = before != after #want to truncate
            met_req = True

            try:
                assert str(pat_row['uniquepid']) == tmp, file+f" mapper wrong {pat_row['uniquepid']} vs {tmp}"
                audit(df2, file)
            except AssertionError as e:
                print()
                print("audit err")
                tocsv=False
                met_req = False
                with open(f"./issues/audit_err{p}.txt", "a") as f:
                    f.write(f"{file}\n")
                    f.write(f"{e}\n\n")

            listfile_tests = {
                'in-hospital-mortality_split/': meet_ihm_criteria,
                'decompensation_split/': meet_decomp_criteria,
                'length-of-stay_split/': meet_los_criteria,
                'phenotyping_split/': meet_pheno_criteria,
            }

            for key in all_patients:
                if file in all_patients[key]:
                    try:
                        ct+=1
                        listfile_tests[key](df2, los, expire_status, unitid, file)
                        assert file in all_patients['phenotyping_split/'], file + "not in phenotype split"
                    except AssertionError as e:
                        print()
                        met_req = False
                        print(key,"listfile err")
                        tocsv=False
                        with open(f"./issues/listfile2_err{p}.txt", "a") as f:
                            f.write(f"{file} {key}\n")
                            f.write(f"{e}\n\n")

            if tocsv:
                with open(f"./issues/truncate{p}.txt", "a") as f:
                    f.write(f"{file} {los}\n")
            
            if met_req:
                with open(f"./issues/visited{p}.txt", "a") as f:
                    f.write(f"{file}\n")

print("DONE!", ct)