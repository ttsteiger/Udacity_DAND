# plotting_functions.py

def scatter_plot(df, x, y, normalize=True):
    """
    
    """
    poi_df = df[df['poi'] == True]
    x_poi = poi_df[x].values.reshape(-1, 1)
    y_poi = poi_df[y].values.reshape(-1, 1)
    
    non_poi_df = df[df['poi'] == False]
    x_non_poi = non_poi_df[x].values.reshape(-1, 1)
    y_non_poi = non_poi_df[y].values.reshape(-1, 1)
    
    if normalize:
        x_poi = MinMaxScaler().fit_transform(x_poi)
        y_poi = MinMaxScaler().fit_transform(y_poi)
        
        x_non_poi = MinMaxScaler().fit_transform(x_non_poi)
        y_non_poi = MinMaxScaler().fit_transform(y_non_poi)
    
    # create plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(x_poi, y_poi, color="red", label="poi")
    ax.scatter(x_non_poi, y_non_poi, color="blue", label="non poi")
    ax.set(title="{} vs. {}".format(x, y), xlabel=x, xlim=[-0.02, 1.02], ylabel=y, ylim=[-0.02, 1.02])
    plt.legend()
    
    plt.show()

scatter_plot(data_df, 'salary', 'bonus')
