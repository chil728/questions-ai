years = [2021, 2022, 2023]
months = [3, 6, 9, 12]
level = 1

for year in years:
    for month in months:
        file = open("Python_%d%02d_1.json"%(year, month), 'w')
        file.close()