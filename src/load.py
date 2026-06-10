def load_data(df, output_path):
    #index = false to remove index as it is not part of the dataframe
    # .to_csv is a built_in pandas function that takes argument and builds it into a csv file

    # in this case, it is .to_csv(where_to_save_the_file, should_i_save_the_index)
    df.to_csv(output_path, index=False)

