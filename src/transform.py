def transform_data(df):

    #removes any duplicates
    df = df.drop_duplicates()

    #removing any row of data if the items in the subset list contain no data
    df = df.dropna(subset = ["track_id", "track_name", "artists", "popularity"])

    #converting milliseconds to minutes
    df["duration_minutes"] = df["duration_ms"] / 60000

    #lower() lowercases everything
    #strip removes any unnecessary indenting such as spaces
    #replace makes it so spaces turn into underscores
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    #removing songs with invalid popularity values
    df = df[(df["popularity"] >= 0) & (df["popularity"] <= 100)]

    #removing orignial index columns if it exists
    if "unnamed:_0" in df.columns:
        df = df.drop(columns = ["unnamed:_0"])

    return df

