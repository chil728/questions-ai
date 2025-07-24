years = [2021, 2022, 2023, 2024]
months = [3, 6, 9, 12]
level = 3

for year in years:
    for month in months:
        file = open("Python_%d%02d_%d.json"%(year, month, level), 'a')
        file.close()