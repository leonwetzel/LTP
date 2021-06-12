import pandas as pd
from sklearn.model_selection import train_test_split

from auxiliary import dividing_dataset

print()
print("-----------")
print()
print("Effect of pre-processing on nr. of instances:")
print()

data_file = "data/PSP_data.csv"

# load dataset
df = pd.read_csv(data_file, sep=',', quotechar='"')
print("Nr. of instances before pre-processing:\t\t\t", df.shape[0])

# clean dataframe from missing values
df = df[df['text'].notnull()]
df = df[df['Category'].notnull()]
print("Nr. of instances after removing null values:\t\t", df.shape[0])

# convert multi class labels to binairy labels
indices = []
for index, row in df.iterrows():
    if "The full tweet text was not retrievable." in row['text']:
       indices.append(index)

# drop rows that contain not retrievable tweets
df = df.drop(indices)
print("Nr. of instances after removing not retrievable tweets:\t", df.shape[0])
print()
print("-----------")
print()

# print nr of comments per country
print("Distribution of comments per country: ")
print()
print(df['Country'].value_counts())
print()
print("Distribution of categories per country (Multi-class labels): ")
print()

total_sexist = 0
total_anti_immigrant = 0
total_anti_muslim = 0
total_anti_semitic = 0
total_homophobic = 0
total_other = 0
total_no = 0
# print distribution of categories per country
# Sexist, Anti-immigrant, Anti-muslim, Anti-semitic, Homophobic, Other, None
print("France:")
sexist = 0
anti_immigrant = 0
anti_muslim = 0
anti_semitic = 0
homophobic = 0
other = 0
no = 0

for instance in df.loc[df['Country'] == 'France']['Category']:
    try:
        for category in instance.split():
            if category == 'Sexist':
                sexist += 1
            if category == 'Anti-immigrant':
                anti_immigrant += 1
            if category == 'Anti-muslim':
                anti_muslim += 1
            if category == 'Anti-semitic':
                anti_semitic += 1
            if category == 'Homophobic':
                homophobic += 1
            if category == 'Other':
                other += 1
            if category == 'None':
                no += 1
    except AttributeError:
        pass

total_sexist += sexist
total_anti_immigrant += anti_immigrant
total_anti_muslim += anti_muslim
total_anti_semitic += anti_semitic
total_homophobic += homophobic
total_other += other
total_no += no

print("    Sexist:", sexist)
print("    Anti-immigrant:", anti_immigrant)
print("    Anti-muslim:", anti_muslim)
print("    Anti-semitic:", anti_semitic)
print("    Homophobic:", homophobic)
print("    Other:", other)
print("    None:", no)
print()


print("Italy:")
sexist = 0
anti_immigrant = 0
anti_muslim = 0
anti_semitic = 0
homophobic = 0
other = 0
no = 0

for instance in df.loc[df['Country'] == 'Italy']['Category']:
    try:
        for category in instance.split():
            if category == 'Sexist':
                sexist += 1
            if category == 'Anti-immigrant':
                anti_immigrant += 1
            if category == 'Anti-muslim':
                anti_muslim += 1
            if category == 'Anti-semitic':
                anti_semitic += 1
            if category == 'Homophobic':
                homophobic += 1
            if category == 'Other':
                other += 1
            if category == 'None':
                no += 1
    except AttributeError:
        pass

total_sexist += sexist
total_anti_immigrant += anti_immigrant
total_anti_muslim += anti_muslim
total_anti_semitic += anti_semitic
total_homophobic += homophobic
total_other += other
total_no += no

print("    Sexist:", sexist)
print("    Anti-immigrant:", anti_immigrant)
print("    Anti-muslim:", anti_muslim)
print("    Anti-semitic:", anti_semitic)
print("    Homophobic:", homophobic)
print("    Other:", other)
print("    None:", no)
print()

print("Germany:")
sexist = 0
anti_immigrant = 0
anti_muslim = 0
anti_semitic = 0
homophobic = 0
other = 0
no = 0

for instance in df.loc[df['Country'] == 'Germany']['Category']:
    try:
        for category in instance.split():
            if category == 'Sexist':
                sexist += 1
            if category == 'Anti-immigrant':
                anti_immigrant += 1
            if category == 'Anti-muslim':
                anti_muslim += 1
            if category == 'Anti-semitic':
                anti_semitic += 1
            if category == 'Homophobic':
                homophobic += 1
            if category == 'Other':
                other += 1
            if category == 'None':
                no += 1
    except AttributeError:
        pass

total_sexist += sexist
total_anti_immigrant += anti_immigrant
total_anti_muslim += anti_muslim
total_anti_semitic += anti_semitic
total_homophobic += homophobic
total_other += other
total_no += no

print("    Sexist:", sexist)
print("    Anti-immigrant:", anti_immigrant)
print("    Anti-muslim:", anti_muslim)
print("    Anti-semitic:", anti_semitic)
print("    Homophobic:", homophobic)
print("    Other:", other)
print("    None:", no)
print()

print("Switzerland:")
sexist = 0
anti_immigrant = 0
anti_muslim = 0
anti_semitic = 0
homophobic = 0
other = 0
no = 0

for instance in df.loc[df['Country'] == 'Switzerland']['Category']:
    try:
        for category in instance.split():
            if category == 'Sexist':
                sexist += 1
            if category == 'Anti-immigrant':
                anti_immigrant += 1
            if category == 'Anti-muslim':
                anti_muslim += 1
            if category == 'Anti-semitic':
                anti_semitic += 1
            if category == 'Homophobic':
                homophobic += 1
            if category == 'Other':
                other += 1
            if category == 'None':
                no += 1
    except AttributeError:
        pass

total_sexist += sexist
total_anti_immigrant += anti_immigrant
total_anti_muslim += anti_muslim
total_anti_semitic += anti_semitic
total_homophobic += homophobic
total_other += other
total_no += no

print("    Sexist:", sexist)
print("    Anti-immigrant:", anti_immigrant)
print("    Anti-muslim:", anti_muslim)
print("    Anti-semitic:", anti_semitic)
print("    Homophobic:", homophobic)
print("    Other:", other)
print("    None:", no)
print()

print("Totals:")
print("    Sexist:", total_sexist)
print("    Anti-immigrant:", total_anti_immigrant)
print("    Anti-muslim:", total_anti_muslim)
print("    Anti-semitic:", total_anti_semitic)
print("    Homophobic:", total_homophobic)
print("    Other:", total_other)
print("    None:", total_no)
print("    Total:", total_sexist+total_sexist+total_anti_muslim+total_anti_semitic+total_homophobic+total_other+total_no)
print()
print("-----------")
print()

print("Distribution of categories per country (Binary labels):")
print()
# convert multi class labels to binairy labels
df.loc[df['Category'] != "None", "Category"] = 'Offensive'
df.loc[df['Category'] == "None", "Category"] = 'Non-offensive'

# print nr of comments per category
print(df['Category'].value_counts())
print()

# print distribution of category per country
print("France:")
print(df.loc[df['Country'] == 'France']['Category'].value_counts())
print()

print("Italy:")
print(df.loc[df['Country'] == 'Italy']['Category'].value_counts())
print()

print("Germany:")
print(df.loc[df['Country'] == 'Germany']['Category'].value_counts())
print()

print("Switzerland:")
print(df.loc[df['Country'] == 'Switzerland']['Category'].value_counts())
print()
