import geopandas as gpd
from shapely.geometry import shape, Point
from shapely.ops import unary_union
import pandas as pd

def process_business(df):
    for col in df.columns: 
        if "date" in col:
            df[col] = pd.to_datetime(df[col])
    to_drop_lst = []
    for col in df.columns:
      if "computed" in col:
        to_drop_lst.append(col)
    df.drop(to_drop_lst, axis=1, inplace=True )
    return df
    
def process_blocks(df):
    df["block_group"] = df["geoid10"].apply(lambda block: block[:12])
    df["the_geom"] = df["the_geom"].apply(shape).apply(unary_union)
    gdf = gpd.GeoDataFrame(df).set_geometry("the_geom").drop(columns=df.columns.difference(["block_group", "the_geom"]))
    gdf = gpd.GeoDataFrame(gdf).set_geometry('the_geom')
    return gdf

def join_with_block_groups(business_df, blocks):
    business_df = business_df.dropna(subset=["longitude", "latitude"])
    business_df["the_geom"] = business_df.apply(lambda row: Point(float(row["longitude"]), float(row["latitude"])), axis=1)
    business_gdf = gpd.GeoDataFrame(business_df).set_geometry("the_geom")
    joined_data = gpd.sjoin(business_gdf, blocks[["block_group", "the_geom"]]).drop(columns="index_right")
    return joined_data

def working(business, tracts, census):
    joined = join_with_block_groups(business, tracts)
    full = pd.merge(joined, census, on='block_group')
    
'''    
new_bus_dict = {}
new_bus_dict['type'] = ['Percentage of New Applications']
prev_year = 2002
for year in year_lst:
    last_year = set(augmented_df[(augmented_df['date_issued'].dt.year == prev_year) & (augmented_df['application_type'] == 'ISSUE')]['account_number'])
    this_year = set(augmented_df[(augmented_df['date_issued'].dt.year == year) & (augmented_df['application_type'] == 'ISSUE')]['account_number'])
    diff = len(this_year) - len(last_year)
    new_bus_dict[year] = [diff*100 / len(last_year)]
    prev_year = year

new_bus = pd.DataFrame.from_dict(new_bus_dict)

renewals_dict = {}
renewals_dict['type'] = ['Percentage of Renewals']
prev_year = 2002
for year in year_lst:
    last_year = set(augmented_df[(augmented_df['date_issued'].dt.year == prev_year) & (augmented_df['application_type'] == 'RENEW')]['account_number'])
    this_year = set(augmented_df[(augmented_df['date_issued'].dt.year == year) & (augmented_df['application_type'] == 'RENEW')]['account_number'])
    diff = len(this_year) - len(last_year)
    renewals_dict[year] = [diff*100 / len(last_year)]
    prev_year = year

renewals = pd.DataFrame.from_dict(renewals_dict)
'''