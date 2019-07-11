# coding: utf-8

import numpy as np
import pandas as pd
import ujson as json
import argparse
import time

# patient_ids = []
#
# for filename in os.listdir('./raw'):
#     # the patient data in PhysioNet contains 6-digits
#     match = re.search('\d{6}', filename)
#     if match:
#         id_ = match.group()
#         patient_ids.append(id_)


parser = argparse.ArgumentParser()
parser.add_argument('--default_path', type=str, default="./")
parser.add_argument('--brits_path', type=str, default="./")
args = parser.parse_args()

default_path = args.default_path
brits_path = args.brits_path

index_column_name = "merchant_name"
value_column_name = "spend"
time_column_name = "month"

# out = pd.read_csv('./raw/Outcomes-a.txt').set_index('RecordID')['In-hospital_death']

# we select 35 attributes which contains enough non-values
# attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2',
#               'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
#               'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
#               'Creatinine', 'ALP']

# mean and std of 35 attributes
# mean = np.array([59.540976152469405, 86.72320413227443, 139.06972964987443, 2.8797765291788986, 58.13833409690321,
#                  147.4835678885565, 12.670222585415166, 7.490957887101613, 2.922874149659863, 394.8899400819931,
#                  141.4867570064675, 96.66380228136883, 37.07362841054398, 505.5576196473552, 2.906465787821709,
#                  23.118951553526724, 27.413004968675743, 19.64795551193981, 2.0277491155660416, 30.692432164676188,
#                  119.60137167841977, 0.5404785381886381, 4.135790642787733, 11.407767149315339, 156.51746031746032,
#                  119.15012244292181, 1.2004983498349853, 80.20321011673151, 7.127188940092161, 40.39875518672199,
#                  191.05877024038804, 116.1171573535279, 77.08923183026529, 1.5052390166989214, 116.77122488658458])
#
# std = np.array(
#     [13.01436781437145, 17.789923096504985, 5.185595006246348, 2.5287518090506755, 15.06074282896952, 85.96290370390257,
#      7.649058756791069, 8.384743923130074, 0.6515057685658769, 1201.033856726966, 67.62249645388543, 3.294112002091972,
#      1.5604879744921516, 1515.362517984297, 5.902070316876287, 4.707600932877377, 23.403743427107095, 5.50914416318306,
#      0.4220051299992514, 5.002058959758486, 23.730556355204214, 0.18634432509312762, 0.706337033602292,
#      3.967579823394297, 45.99491531484596, 21.97610723063014, 2.716532297586456, 16.232515568438338, 9.754483687298688,
#      9.062327978713556, 106.50939503021543, 170.65318497610315, 14.856134327604906, 1.6369529387005546,
#      133.96778334724377])

fs = open(brits_path + 'json/json', 'w')


def to_time_bin(x):
    h, m = map(int, x.split(':'))
    return h


def to_month_bin(x):
    y, m, d = map(int, x.split('-'))
    return y * 12 + (m - 1)


def parse_data(x):
    x = x.set_index(index_column_name).to_dict()[value_column_name]

    values = []

    for attr in attributes:
        if attr in x:
            values.append(x[attr])
        else:
            values.append(np.nan)
    return values


def parse_delta(masks, dir_, num_rows, num_features):
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []

    for h in range(num_rows):
        if h == 0:
            deltas.append(np.ones(num_features))
        else:
            deltas.append(np.ones(num_features) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)


def parse_rec(values, masks, evals, eval_masks, dir_, num_rows, num_features):
    deltas = parse_delta(masks, dir_, num_rows, num_features)

    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).as_matrix()

    rec = {}

    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()

    return rec


def parse_id(data, min_date, max_date):

    evals = []

    not_for_train = False

    # merge all the metrics within one month
    for m in range(min_date, max_date + 1):
        if len(data[data['month'] == m]) == 0:
            print("missed data for {}".format(m))
            not_for_train = True
            break
        evals.append(parse_data(data[data['month'] == m]))

    # if user has missed data we exclude him from train dataset
    if not_for_train:
        return False

    # normalization
    # evals = (np.array(evals) - mean) / std

    evals = np.array(evals)

    shp = evals.shape

    evals = evals.reshape(-1)

    # randomly eliminate 10% values as the imputation ground-truth
    indices = np.where(~np.isnan(evals))[0].tolist()
    indices = np.random.choice(indices, len(indices) // 10)

    values = evals.copy()
    values[indices] = np.nan

    masks = ~np.isnan(values)
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))

    evals = evals.reshape(shp)
    values = values.reshape(shp)

    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)

    rec = {'label': 1}

    num_rows = max_date - min_date + 1

    # prepare the model for both directions
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, 'forward',
                               num_rows, len(attributes))
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], 'backward',
                                num_rows, len(attributes))

    rec = json.dumps(rec)

    fs.write(rec + '\n')

    return True


gaps = pd.read_csv(default_path + 'gaps.csv')

shops = gaps['merchant_name'].unique().tolist()
ids = gaps['unique_mem_id'].unique().astype('int64').tolist()

# accumulate the records within one hour
gaps['month'] = gaps['month'].apply(lambda x: to_month_bin(x))
min_date = gaps['month'].min()
max_date = gaps['month'].max()

attributes = shops

i = 0
train_num = 0
missed_num = 0
total = len(ids)
for id_ in ids:
    i += 1
    print('Processing patient {} ({}/{})'.format(id_, i, total))

    s = time.time()

    try:

        data = gaps[gaps['unique_mem_id'] == id_]
        is_for_train = parse_id(data, min_date, max_date)
        if is_for_train:
            train_num += 1
        else:
            missed_num += 1
    except Exception as e:
        print(e)
        continue

    e = time.time()
    print('elapsed time: {}, train dataset len: {}, missed dataset len: {}'.format(e - s, train_num, missed_num))

fs.close()

