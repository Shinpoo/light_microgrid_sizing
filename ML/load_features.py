import pandas as pd

path = "data/load.csv"
df = pd.read_csv(path, sep=";|,", parse_dates=True, index_col='DateTime', engine='python')
companies = df.columns.to_list()
print(companies)
df['Year'] = df.index.map(lambda x: x.year)
df['Month'] = df.index.map(lambda x: x.month)
df['Day'] = df.index.map(lambda x: x.day)
df['Hour'] = df.index.map(lambda x: x.hour)
df['Minutes'] = df.index.map(lambda x: x.minute)
df['Seconds'] = df.index.map(lambda x: x.second)
df['IsoDayOfWeek'] = df.index.map(lambda x: x.isoweekday())
df['Season'] = df.index.map(lambda x: x.month % 12 // 3 + 1)
# print(df)
# #Mean load for each month
# print(df['ANBRIMEX'].mean())
# df_season_mean = df.groupby('Season')['ANBRIMEX'].mean()
# df_season_min = df.groupby('Season')['ANBRIMEX'].min()
# df_season_med = df.groupby('Season')['ANBRIMEX'].median()
# winter_df = df.loc[df['Season'] == 1]
# spring_df = df.loc[df['Season'] == 2]
# summer_df = df.loc[df['Season'] == 3]
# autumn_df = df.loc[df['Season'] == 4]
# winter_df.nlargest(5, ['ANBRIMEX'])
# Get the mean of the 20% highest loads of ANBRIMEX during winter:
mean_peaks = dict.fromkeys(companies)
mean_bases = dict.fromkeys(companies)
for c in companies:
    mean_peak_season = []
    mean_base_season = []
    for season in range(1, 5):
        print(c, season)
        mean_peak = df.loc[df['Season'] == season].nlargest(int(len(df.loc[df['Season'] == season]) * 0.2), [c])[
            c].mean()
        mean_base = df.loc[df['Season'] == season].nsmallest(int(len(df.loc[df['Season'] == season]) * 0.2), [c])[
            c].mean()
        mean_peak_season.append(mean_peak)
        mean_base_season.append(mean_base)
        print(mean_base)
        print(mean_peak)
    mean_peaks[c] = sum(mean_peak_season) / len(mean_peak_season)
    mean_bases[c] = sum(mean_base_season) / len(mean_base_season)

print(mean_peaks)
print(mean_bases)
case_name = "ANBRIMEX"
df = pd.read_csv("results/" + case_name + "_with_loc.csv", sep=";|,", engine="python", index_col='index')
df.loc[df.name == "ANBRIMEX", 'avg_peak'] = mean_peaks['ANBRIMEX']
df.loc[df.name == "ANBRIMEX", 'avg_base'] = mean_bases['ANBRIMEX']

df.to_csv("results/" + case_name + "_final.csv", sep=';',
          encoding='utf-8', index=True, decimal=".")
