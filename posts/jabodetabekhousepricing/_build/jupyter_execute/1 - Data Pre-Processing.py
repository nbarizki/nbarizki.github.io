#!/usr/bin/env python
# coding: utf-8

# # **Data Pre-Processing**

# ## 1. Introduction

# ## 2. Combining Dataset of All Jabodetabek

# In[20]:


import pandas as pd
import numpy as np
from numpy import vectorize
import ast


# In[21]:


file_names = [
    'Scraped_Data/raw/Bekasi_2022-10-09_16-57_page_1_to_50.csv',
    'Scraped_Data/raw/Bogor_2022-10-08_17-49_page_1_to_20.csv',
    'Scraped_Data/raw/Bogor_2022-10-08_18-19_page_21_to_30.csv',
    'Scraped_Data/raw/Bogor_2022-10-09_18-20_page_31_to_50.csv',
    'Scraped_Data/raw/Depok_2022-10-09_17-17_page_1_to_21.csv',
    'Scraped_Data/raw/Depok_2022-10-09_14-23_page_22_to_50.csv',
    'Scraped_Data/raw/Jakarta_2022-10-07_08-14_page_1_to_10.csv',
    'Scraped_Data/raw/Jakarta_2022-10-08_13-34_page_11_to_20.csv',
    'Scraped_Data/raw/Jakarta_2022-10-09_17-20_page_21_to_37.csv',
    'Scraped_Data/raw/Jakarta_2022-10-08_15-00_page_38_to_40.csv',
    'Scraped_Data/raw/Jakarta_2022-10-08_15-41_page_41_to_50.csv',
    'Scraped_Data/raw/Tangerang_2022-10-08_21-10_page_1_to_50.csv'
]

df = pd.concat(
    map(pd.read_csv, file_names), ignore_index=True
)

df.sample(2).T


# ## 3. Column Field Correction

# As we can see, there are inconsistence column labels due to different language format. Let's compare both condition:

# In[22]:


list((df[~(df['k. tidur'].isna())].sample(1).index.values,
        df[~(df['bedrooms'].isna())].sample(1).index.values)
)


# In[23]:


sample = pd.concat([
        df[(df['k. tidur'].notnull())].sample(1),
        df[(df['bedrooms'].notnull())].sample(1)
])

sample.T


# In[24]:


row_count = pd.DataFrame({
    'row_count' : [
        df[df['k. tidur'].notnull()].shape[0],
        df[df['bedrooms'].notnull()].shape[0]
        ]
    })
row_count.index=['ind_labeled', 'eng_labeled']
row_count


# As we can see, where `Indonesian labeled column` is recorded, the `English labeled column` is empty, and vice versa. I decide to keep the English label using codes below:

# In[25]:


label_pair = [
    ('k. tidur', 'bedrooms'),
    ('k. mandi', 'bathrooms'),
    ('l. tanah', 'land size'),
    ('l. bangunan', 'building size'),
    ('sertifikat', 'certificate'),
    ('daya listrik', 'electricity'),
    ('km. pembantu', 'maid bathrooms'),
    ('jumlah lantai', 'floors'),
    ('tahun dibangun', 'building age'),
    ('kondisi properti', 'property condition'),
    ('kondisi perabotan', 'furnishing'),
    ('carport', 'carports'),
    ('hadap', 'building orientation'),
    ('garasi', 'garages'),
    ('kt. pembantu', 'maid bedrooms')
]


# We will use `DataFrame.fillna()` (column-wise) to fill missing records of `English labeled columns` using `Indonesian labeled columns` value as below:

# In[26]:


for ind, eng in label_pair:
    df[eng].fillna(value=df[ind], inplace=True)
df.sample(2).T


# Looks like it turns out as we expected. Now we just drop the `Indonesian labeled columns` using below codes. We also rename the columns that still in Indonesian label and using `snake_case` format:

# In[27]:


for column in label_pair:
    df.drop(labels=column[0], axis=1, inplace=True)

df.rename(
    columns={
        'tipe properti': 'property_type',
        'id iklan': 'ads_id'
        }, 
    inplace=True
)

df.rename(
    columns={
        column: column.replace(' ', '_') for column in df.columns
    },
    inplace=True
)

df.sample(2).T


# ## 4. Transforming Price and Address Records

# #### **Price Records**

# We are going to check unique values of `currency` and `price_unit_scale`:

# In[28]:


print(df.currency.unique())
print(df.price_unit_scale.unique())


# We should check the house price in `Triliun`, since the value is rather ambiguous for a house price to be this high.

# In[29]:


df[df.price_unit_scale=='Triliun']


# After checking the URL, seems that price is in Rp. 5,4 Miliar (listing may be revised by owner) so we shall revise the price scale for this record.

# In[30]:


df.loc[df.price_unit_scale=='Triliun', 'price_unit_scale'] = 'Miliar'
print(df.price_unit_scale.unique())


# The price is still in `string` format, so we will transform the value by its `price_unit_scale`.
# 
# Pandas need `vectorized function` to execute `vectorized operations`. We can create `vectorized function` by creating a standard function first then we convert it into vectorized function using `np.vectorize`. Note that `np.vectorize` returns a `callable` as its output, which we will use to our Pandas operations.

# In[31]:


def convert_price(price: str, unit_scale: str):
    price_numeric = float(price.replace(',', '.'))
    if unit_scale == 'Juta':
        converted_price = price_numeric * 1000000
    else:
        converted_price = price_numeric * 1000000000
    return converted_price

convert_price_vectd = vectorize(convert_price)


# Let's try our `vectorized function` to our dataframe. It is convenient to change panda's setting of number display format so that we can clearly distinguish the millions and billions scale.

# In[32]:


pd.set_option('display.float_format', '{:,.2f}'.format)

df.assign(
    price_converted=convert_price_vectd(df.price, df.price_unit_scale)
).loc[:, ['currency', 'price', 'price_unit_scale', 'price_converted']].sample(3, random_state=4).T


# As expected, `Pandas` performing vectorized function, resulting in an index-wise operation to convert the price. The scale of `Miliar` and `Juta` is also correct as expected.
# 
# We won't need the `price_unit_scale` anymore, and we will define a new column `price_in_rp` to inform that the currency is in `Rupiahs`.

# In[33]:


df = df.assign(
    price = convert_price_vectd(df.price, df.price_unit_scale)
    ).rename(
        columns={'price': 'price_in_rp'}
    ).drop(
        ['currency', 'price_unit_scale'], axis=1
    )

df.sample(2).T


# #### **Address**

# Address records contains (`district`, `city`). Splitting into this may be useful when reporting insight, so I will also provide the separate value of this.

# In[34]:


def get_district(address:str):
    return address.strip().split(sep=',')[0]

def get_city(address:str):
    return address.strip().split(sep=',')[1]

get_district_vectd = vectorize(get_district)
get_city_vectd = vectorize(get_city)


# In[35]:


df.assign(
    district=get_district_vectd(df.address),
    city=get_city_vectd(df.address)
).loc[:, ['address', 'district', 'city']].sample(3).T


# Results is as expected. We will apply this to our Dataframe, and positions the `district` and `city` next to the `address` for convenience.

# In[36]:


columns_pair = [('city', get_city_vectd), ('district', get_district_vectd)]
for (col, func) in columns_pair:
    df.insert(4, col, func(df.address))

df.sample(2).T


# In notebook `1. Web Scraping`, we also scrap `estimated latitude and longitude` of each district, because the primary website doesn't provide any coordinate information.
# 
# We are going to join the `latitude and longitude` data using below codes. `Adress` column will be used as the `join key`.

# In[37]:


lat_long_df = pd.read_csv('Scraped_Data/lat_long_complete.csv')
lat_long_df.head()


# In[38]:


def split_lat_long(value):
    lat_long = ast.literal_eval(value)
    return lat_long

split_lat_long_vectd = vectorize(split_lat_long)
lat_long_df = lat_long_df.assign(
    lat=split_lat_long_vectd(lat_long_df.lat_long)[0],
    long=split_lat_long_vectd(lat_long_df.lat_long)[1],
)
lat_long_df.head()


# In[39]:


# merge dataframes
df = df.merge(
    right=lat_long_df.loc[:, ['address', 'lat', 'long']],
    on='address'
)
# re-arrange columns
cols = df.columns.to_list()
cols.insert(6, 'long')
cols.insert(6, 'lat')
cols_arranged = cols.copy()[:-2]
df = df[cols_arranged]

df.sample(2).T


# #### **Land Size, Building Size, Electricity**

# Records of these columns is still in `string` format as (`value`, `unit`). We will extract only the number and rename the columns to inform the unit.

# In[40]:


def get_value(value:str):
    if (type(value) == str):
        return float(value.strip().split(sep=' ')[0])
    else:
        return float(value)
    
get_value_vectd = vectorize(get_value)


# In[41]:


df = df.assign(
    land_size=get_value_vectd(df.land_size),
    building_size=get_value_vectd(df.building_size),
    ).rename(
        columns={
            'land_size': 'land_size_m2',
            'building_size': 'building_size_m2'
        }
    )

df.sample(2).T


# Let's inspect the `electricity`. The category of electricity power for residential in Indonesia should not varied vastly.

# In[42]:


df.electricity.unique()


# Later we can replace the missing `nan` value with `lainnya mah`. And also, the electricity is naturally has an `ordinal` order, which must be considered in later analysis.

# #### **Building Age**

# Let's inspect the `Building Age` columns:

# In[43]:


df.building_age.unique()


# Some concerns:
# 
# 1. Based on its values, it is best to be explained as `year_built`. We will rename the columns.
# 2. Unfortunately we don't find the `date listed` information in the listing info since it is best to explain *what owner think their house priced in the year that the house is listed and also considering house age since first it been built*. But it is okay to assumme that at the time the listing is still advertised in the website then owner *did* consider the value adhere to current year valuation. So we can extract `building_age = year_built - 2022`.  
# 3. There are some values that ambiguous: `2052, 20010, 20012, 2025`. For values `2025`, it may be because house is available after future housing project will be completed at `2025` (but this justification is still not make sense for `2052`) 

# In[44]:


df.rename(
    columns={'building_age': 'year_built'},
    inplace=True
)


# In[45]:


years = [2052, 20010, 20012, 2025]
df[df.year_built.isin(years)].T


# After inspecting each of the records, it is safe to say that:
# 
# 1. For `year_built > 2022`, let's just keep the value as it is, since it is better be explained as `building age = 0`.
# 2. For `year_built in [20010, 20012]`, based on `property_condition`, let's assume best case scenario to be `2010 and 2012`.

# In[46]:


def building_age(year_built):
    if year_built <= 2022:
        return (2022 - year_built)
    elif year_built > 2022:
        return 0
    return year_built

building_age_vectd = vectorize(building_age)
df.loc[df.year_built == 20010, 'year_built'] = 2010
df.loc[df.year_built == 20012, 'year_built'] = 2012
df.insert(21, 'building_age', building_age_vectd(df.year_built))

df.building_age.unique()


# ## 5. Considering the Missing Value

# Missing values for column fields explained below can naturally be considired as `not provided` so can be replaced by `0` (`lainnya mah` for missing electricity type):
# 
# 1. Maid Bedrooms
# 2. Maid Bathrooms
# 3. Carports
# 4. Garages
# 5. Electricity

# In[47]:


fill_columns = ['maid_bedrooms', 'maid_bathrooms', 'carports', 'garages']

for column in fill_columns:
    df[column].fillna(0, inplace=True)

df['electricity'].fillna('lainnya mah', inplace=True)
df[
    df.maid_bedrooms.isna() | df.maid_bathrooms.isna() | df.carports.isna() | df.garages.isna() | df.electricity.isna()
    ]


# Other than aboves column fields, missing value is considered missing information (since it explains the essential characteristic of the house) and will be evaluated in later analysis.

# ## 6. Pre-Processed Data

# Our preprocessed data looks like below sample:

# In[48]:


df[~df.isna()].sample(2).T


# In[49]:


df.to_csv('Scraped_Data/jabodetabek_house_price.csv', index=False)

