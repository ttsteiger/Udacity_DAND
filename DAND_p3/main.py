import csv
import matplotlib.pyplot as plt
import mplleaflet
import os
import pprint
import re
import sqlite3
import xml.etree.cElementTree as ET

import audit
import clean
import explore
import shape

file = os.path.normpath("data/zurich.osm")

## data exploration
root_attributes, child_attributes = explore.get_file_structure(file)

print("Root elements and their attributes:")
pprint.pprint(root_attributes)
print()

print("Children and their attributes within the root elements:")
pprint.pprint(child_attributes)
print()

tag_keys = data_exploration.get_tag_keys(file, ['node', 'way', 'relation'])
way_tag_surface_values = data_exploration.get_tag_key_values(file, ['way'], 'surface')

# print out top 20 entries
print("Most frequently occuring tag keys:")
pprint.pprint(tag_keys[:20])
print()
print("Most frequently occuring ")
pprint.pprint(way_tag_surface_values[:20])

## data auditing
bad_house_numbers = []
bad_cities = []
bad_phone_numbers = []
bad_websites = []

audit.audit(file, bad_house_numbers, bad_cities, bad_phone_numbers, bad_websites)

print("Audited house numbers:")
pprint.pprint(bad_house_numbers[:20])
print()
print("Audited cities:")
pprint.pprint(bad_cities[:20])
print()
print("Audited phone numbers:")
pprint.pprint(bad_phone_numbers[:20])
print()
print("Audited websites:")
pprint.pprint(bad_websites[:20])
print()

## SQL database
# keys for the dictionaries to be shaped for the different root elements
node_fields = ['node_id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
node_tag_fields = ['node_id', 'key', 'value', 'type']

way_fields = ['way_id', 'user', 'uid', 'version', 'changeset', 'timestamp']
way_tag_fields = ['way_id', 'key', 'value', 'type']
way_node_fields = ['way_id', 'node_id', 'position']

relation_fields = ['relation_id', 'user', 'uid', 'version', 'changeset', 'timestamp']
relation_tag_fields = ['relation_id', 'key', 'value', 'type']
relation_member_fields = ['relation_id', 'type', 'node_id', 'way_id', 'role']

# regex for characters we do not want in the tag keys
problem_chars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

# csv file paths
nodes_path = os.path.normpath("data/nodes.csv")
nodes_tags_path = os.path.normpath("data/nodes_tags.csv")

ways_path = os.path.normpath("data/ways.csv")
ways_tags_path = os.path.normpath("data/ways_tags.csv")
ways_nodes_path = os.path.normpath("data/ways_nodes.csv")

relations_path = os.path.normpath("data/relations.csv")
relations_tags_path = os.path.normpath("data/relations_tags.csv")
relations_members_path = os.path.normpath("data/relations_members.csv")

# write to csv
def convert_to_csv(file_name):
    with open(nodes_path, 'w', encoding='utf-8') as nodes_file, \
         open(nodes_tags_path, 'w', encoding='utf-8') as nodes_tags_file, \
         open(ways_path, 'w', encoding='utf-8') as ways_file, \
         open(ways_tags_path, 'w', encoding='utf-8') as ways_tags_file, \
         open(ways_nodes_path, 'w', encoding='utf-8') as ways_nodes_file, \
         open(relations_path, 'w', encoding='utf-8') as relations_file, \
         open(relations_tags_path, 'w', encoding='utf-8') as relations_tags_file, \
         open(relations_members_path, 'w', encoding='utf-8') as relations_members_file:
        
        # set up the csv dictwriters
        nodes_writer = csv.DictWriter(nodes_file, node_fields)
        nodes_tags_writer = csv.DictWriter(nodes_tags_file, node_tag_fields)
        
        ways_writer = csv.DictWriter(ways_file, way_fields)
        ways_tags_writer = csv.DictWriter(ways_tags_file, way_tag_fields)
        ways_nodes_writer = csv.DictWriter(ways_nodes_file, way_node_fields)
        
        relations_writer = csv.DictWriter(relations_file, relation_fields)
        relations_tags_writer = csv.DictWriter(relations_tags_file, relation_tag_fields)
        relations_members_writer = csv.DictWriter(relations_members_file, relation_member_fields)
        
        # write headers containing the field names to the csv files
        nodes_writer.writeheader()
        nodes_tags_writer.writeheader()
        
        ways_writer.writeheader()
        ways_tags_writer.writeheader()
        ways_nodes_writer.writeheader()
        
        relations_writer.writeheader()
        relations_tags_writer.writeheader()
        relations_members_writer.writeheader()
        
        # iterate trough all elements in the input file
        for event, elem in ET.iterparse(file_name, events=('end',)):
            # shape element
            shaped_elem = shape.shape_element(elem, node_attr_fields=node_fields, way_attr_fields=way_fields, 
                                              relation_attr_fields=relation_fields, 
                                              problem_chars=problem_chars)
            
            # write element to proper csv file
            if elem.tag == 'node':
                nodes_writer.writerow(shaped_elem['node'])
                nodes_tags_writer.writerows(shaped_elem['node_tags'])
            elif elem.tag == 'way':
                ways_writer.writerow(shaped_elem['way'])
                ways_tags_writer.writerows(shaped_elem['way_tags'])
                ways_nodes_writer.writerows(shaped_elem['way_nodes'])
            elif elem.tag == 'relation':
                relations_writer.writerow(shaped_elem['relation'])
                relations_tags_writer.writerows(shaped_elem['relation_tags'])
                relations_members_writer.writerows(shaped_elem['relation_members'])
        
        print("All files were created successfully!")

convert_to_csv(file)


def create_table(conn, create_table_sql):
    """Create new sql table based on the command given as string."""
    c = conn.cursor()
    c.execute(create_table_sql)


# SQL commands
create_nodes_table_sql = """CREATE TABLE IF NOT EXISTS nodes (
                                node_id INTEGER PRIMARY KEY NOT NULL,
                                lat REAL,
                                lon REAL,
                                user TEXT,
                                uid INTEGER,
                                version INTEGER,
                                changeset INTEGER,
                                timestamp TEXT
                            );"""

create_nodes_tags_table_sql = """CREATE TABLE IF NOT EXISTS nodes_tags (
                                     node_id INTEGER,
                                     key TEXT,
                                     value TEXT,
                                     type TEXT,
                                     FOREIGN KEY (node_id) REFERENCES nodes(id)
                                 );"""

create_ways_table_sql = """CREATE TABLE IF NOT EXISTS ways (
                               way_id INTEGER PRIMARY KEY NOT NULL,
                               user TEXT,
                               uid INTEGER,
                               version INTEGER,
                               changeset INTEGER,
                               timestamp TEXT
                           );"""

create_ways_tags_table_sql = """CREATE TABLE IF NOT EXISTS ways_tags (
                                    way_id INTEGER,
                                    key TEXT,
                                    value TEXT,
                                    type TEXT,
                                    FOREIGN KEY (way_id) REFERENCES ways(id)
                                );"""

create_ways_nodes_table_sql = """CREATE TABLE IF NOT EXISTS ways_nodes (
                                     way_id INTEGER,
                                     node_id INTEGER,
                                     position INTEGER,
                                     FOREIGN KEY (way_id) REFERENCES ways(id),
                                     FOREIGN KEY (node_id) REFERENCES nodes(id)
                                 );"""
create_relations_table_sql = """CREATE TABLE IF NOT EXISTS relations (
                                    relation_id INTEGER PRIMARY KEY NOT NULL,
                                    user TEXT,
                                    uid INTEGER,
                                    version INTEGER,
                                    changeset INTEGER,
                                    timestamp TEXT
                                );"""

create_relations_tags_table_sql = """CREATE TABLE IF NOT EXISTS relations_tags (
                                         relation_id INTEGER,
                                         key TEXT,
                                         value TEXT,
                                         type TEXT,
                                         FOREIGN KEY (relation_id) REFERENCES relations(id)
                                     );"""

create_relations_members_table_sql = """CREATE TABLE IF NOT EXISTS relations_members (
                                            relation_id INTEGER,
                                            type TEXT,
                                            node_id INTEGER,
                                            way_id INTEGER,
                                            role TEXT,
                                            FOREIGN KEY (relation_id) REFERENCES relations(id),
                                            FOREIGN KEY (node_id) REFERENCES nodes(id),
                                            FOREIGN KEY (way_id) REFERENCES ways(id)
                                        );"""

# list containing all querries to create the tables
sql_tables = [create_nodes_table_sql, create_nodes_tags_table_sql,
              create_ways_table_sql, create_ways_tags_table_sql, create_ways_nodes_table_sql,
              create_relations_table_sql, create_relations_tags_table_sql, create_relations_members_table_sql]

# db name
db = "zurich.db"

# create database and set up the tables
conn = sqlite3.connect(db) # creates new db if it does not already exist
for t in sql_tables:
    create_table(conn, t)

conn.close()


def import_csv_file_into_db(db_name, file_name, table_name):
    """
    Import csv file into database. The table needs to exist already.
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        f_reader = csv.DictReader(f)
        header = f_reader.fieldnames
        
        # list of lists containing the csv rows for the db import
        db_data = [[r[k] for k in header] for r in f_reader]
    
    # construct sql command
    columns = "({})".format(', '.join(header)) # concatenate field names
    values = "({})".format(', '.join(['?' for i in range(len(header))])) # number of ? equal to number of fields
    sql_command = "INSERT INTO {} {} VALUES {};".format(table_name, columns, values)
    
    # connect to db and import the data
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    cursor.executemany(sql_command, db_data)
    conn.commit()
    
    conn.close()
    
    print("Import of {} successful!".format(file_name))

# list containing all csv file paths
files = [nodes_path, nodes_tags_path, 
         ways_path, ways_tags_path, ways_nodes_path,
         relations_path, relations_tags_path, relations_members_path]

# get table names for the import function
conn = sqlite3.connect(db)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [t[0] for t in cursor.fetchall()]

conn.close()

# write each csv file to its correpsonding table
for f, t in zip(files, tables):
    import_csv_file_into_db(db, f, t)

## additional data analysis

# restaurants and their cuisines
# sql query
sql_query = """
SELECT lon, lat
FROM nodes, nodes_tags
WHERE nodes_tags.node_id = nodes.node_id
AND nodes_tags.key="amenity"
AND nodes_tags.value="restaurant";
"""

conn = sqlite3.connect('zurich.db')
cursor = conn.cursor()

cursor.execute(sql_query)
results = cursor.fetchall()

conn.close()

# extract lon and lat
lon, lat = [x for x, y in results], [y for x, y in results]

# plot restaurant positions on mplleaflet map
fig1 = plt.figure(figsize=(15, 8))

plt.scatter(lon, lat)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
mplleaflet.show()

# sql query
sql_query = """
SELECT nodes_tags.value, count(*) AS num
FROM nodes, nodes_tags
WHERE nodes_tags.node_id = nodes.node_id
AND nodes_tags.key="cuisine"
GROUP BY nodes_tags.value
ORDER BY num DESC
LIMIT 10;
"""

conn = sqlite3.connect('zurich.db')
cursor = conn.cursor()

cursor.execute(sql_query)
results = cursor.fetchall()

conn.close()

# index, labels and counts for the bar plot
ind = range(len(results))
l, c = [l for l, c in results], [c for l, c in results]

fig2, ax = plt.subplots(figsize=(8, 6))

ax.bar(ind, c)
ax.set(title="Most popular Cuisines in Zürich", ylabel="Number of Restaurants", xticks=ind)
ax.set_xticklabels(l, rotation=45)
ax.grid(alpha=0.4, axis='y')

# sports
# sql query
sql_query = """
SELECT value, count(*) AS num
FROM (SELECT value FROM nodes_tags WHERE key="sport"
      UNION ALL SELECT value FROM ways_tags WHERE key="sport"
      UNION ALL SELECT value FROM relations_tags WHERE key="sport")
GROUP BY value
ORDER BY num DESC
LIMIT 20;
"""

conn = sqlite3.connect('zurich.db')
cursor = conn.cursor()

cursor.execute(sql_query)
results = cursor.fetchall()

conn.close()

# index, labels and counts for the horizontal bar plot
ind = range(len(results))
l, c = [l for l, c in results[::-1]], [c for l, c in results[::-1]]

fig3, ax = plt.subplots(figsize=(8, 6))

ax.barh(ind, c)
ax.set(title="Most popular Sports in Zürich", ylabel="Number of Sport Tags", yticks=ind)
ax.set_yticklabels(l, rotation=0)
ax.grid(alpha=0.4, axis='x')

plt.show()
