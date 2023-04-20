# Build function to import mapping
def import_data(selection_id, mapping_id, gc_input):
  
  # Open our new sheet and add some data.
  ws_1 = gc_input.open_by_url('https://docs.google.com/spreadsheets/d/'+selection_id+'/edit#gid=0').get_worksheet(0)
  ws_2 = gc_input.open_by_url('https://docs.google.com/spreadsheets/d/'+mapping_id+'/edit#gid=0').get_worksheet(1)

  # Create data frames
  df = pd.DataFrame(columns=ws_1.get_all_values()[0], data=ws_1.get_all_values()[1:])
  mapping = pd.DataFrame(columns=ws_2.get_all_values()[0], data=ws_2.get_all_values()[1:])

  # Combine the two tables and pivot
  df_combined = pd.merge(df, mapping, left_on = ['Submission'], right_on = ['ID'], how = 'inner').loc[:,['Name', 'Pick_1', 'Pick_2', 'Pick_3']]
  df_pivot = pd.melt(df_combined, id_vars=['Name'])

  # produce the pivoted df from this function
  return df_pivot


# Build function to run the code
def run_project_cupid(gc):

  # Authenticate
  auth.authenticate_user()
  creds, _ = default()
  gc = gspread.authorize(creds)

  # Get the URLs to pull in
  cookbook = gc.open_by_url('https://docs.google.com/spreadsheets/d/13j_eRYzmCKC4KT8wpKDNw8aefJ1uK4-GvDGBLeBcMhQ/edit#gid=0').get_worksheet(0)
  cookbook_df = pd.DataFrame(columns=cookbook.get_all_values()[0], data=cookbook.get_all_values()[1:])

  # Import data
  df_pivot = import_data(selection_id=cookbook_df['Preferences'][0], mapping_id=cookbook_df['Mappings'][0], gc_input=gc)

  # Inner join the table on itself to return matches
  final_matches = pd.merge(df_pivot, 
                        df_pivot, 
                        left_on=['Name', 'value'], 
                        right_on=['value', 'Name'])

  # Clean the data and rename the columns
  final_matches_clean = (
      final_matches
      .loc[final_matches['Name_x'] != final_matches['value_x'],['Name_x', 'value_x']]
      .sort_values(by = ['Name_x'])
      .rename(columns={'Name_x': 'Person 1', 'value_x': 'Person 2'})
      ).drop_duplicates()

  # Print final results
  return print(final_matches_clean.sort_values(by=['Person 1']))

# Define the function
def create_network_map(show_names=False, gc):

  # Get the URLs to pull in
  cookbook = gc.open_by_url('https://docs.google.com/spreadsheets/d/13j_eRYzmCKC4KT8wpKDNw8aefJ1uK4-GvDGBLeBcMhQ/edit#gid=0').get_worksheet(0)
  cookbook_df = pd.DataFrame(columns=cookbook.get_all_values()[0], data=cookbook.get_all_values()[1:])

  # Create network map df
  df_nm = df_pivot.rename(columns={'Name':'source', 'value':'target'}).loc[:,['source', 'target']]

  # create an empty directed graph
  G = nx.DiGraph()

  # add edges from pandas DataFrame to the graph
  for _, edge in df_nm.iterrows():
      G.add_edge(edge['source'], edge['target'])

  # draw the graph
  plt.figure(figsize=(16, 10))
  pos = nx.spectral_layout(G)
  nx.draw_networkx_nodes(G, pos, node_size=150)
  nx.draw_networkx_edges(G, pos, width=0.2)
  if show_names == True:
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
  plt.axis('off')
  plt.show()
