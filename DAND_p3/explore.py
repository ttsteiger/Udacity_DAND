import xml.etree.cElementTree as ET

def get_roots(file_name, root_tags=['node', 'way', 'relation']):
    """Return a dictionary containing the different specified root elements and their number of occurences in the input file."""
    roots = {}
    for event, elem in ET.iterparse(file_name, events=('end',)):
        if elem.tag in root_tags:
            if elem.tag in roots:
                roots[elem.tag] += 1
            else:
                roots[elem.tag] = 1
            
    return roots


def get_childs(file_name, root):
    """Return a dictionary containing the different childs and their number of occurences for the specified root element."""
    childs = {}
    for event, elem in ET.iterparse(file_name, events=('end',)):
        if elem.tag == root:
            for child in elem:
                if child.tag in childs:
                    childs[child.tag] += 1
                else:
                    childs[child.tag] = 1
    
    return childs


def get_attributes(file_name, root, child_tag=None):
    """
    Return the attributes and their number of occurences for the root element. If child_tag is specified, return the attributes 
    of the child element.
    """
    attributes = {}
    for event, elem in ET.iterparse(file_name, events=('end',)):
        if elem.tag == root:
            # get attributes of child
            if child_tag:
                for child in elem.iter(child_tag):
                    for k in child.keys():
                        if k in attributes:
                            attributes[k] += 1
                        else:
                            attributes[k] = 1
            # get attributes of root  
            else:
                for k in elem.keys():
                    if k in attributes:
                        attributes[k] += 1
                    else:
                        attributes[k] = 1
    
    return attributes


def get_file_structure(file_name):
    """
    Count the root and child elements and their tags for the xml file with 2 levels. 
    """
    # get root element counts
    roots = get_roots(file_name)

    # get children for all root elements
    childs = {}
    for r in roots.keys():
        childs[r] = get_childs(file_name, r)
    
    # collect attributes for all root elements
    root_attributes = {}
    for r in roots.keys():
        root_attributes[r] = get_attributes(file_name, r)
    
    # gather attributes for all the differen child elements
    child_attributes = {}
    for r in roots.keys():
        child_attributes[r] = {}
        for c in childs[r].keys():
            child_attributes[r][c] = get_attributes(file_name, r, c)
    
    return (root_attributes, child_attributes)


def get_tag_keys(file_name, roots):
    tag_keys = {}
    for event, elem in ET.iterparse(file_name, events=('end',)):
        if elem.tag in roots:
            for tag in elem.iter('tag'):
                k = tag.attrib['k']
                if k in tag_keys:
                    tag_keys[k] += 1
                else:
                    tag_keys[k] = 1
    
    # sort tag keys based on numer of occureces
    sorted_tag_keys = sorted(tag_keys.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_tag_keys


def get_tag_key_values(file_name, roots, tag_key):
    tag_key_values = {}
    for event, elem in ET.iterparse(file_name, events=('end',)):
        if elem.tag in roots:
            for tag in elem.iter('tag'):
                if tag.attrib['k'] == tag_key:
                    v = tag.attrib['v']
                    if v in tag_key_values:
                        tag_key_values[v] += 1
                    else:
                        tag_key_values[v] = 1
    
    
    # sort tag key values based on numer of occureces
    sorted_tag_key_values = sorted(tag_key_values.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_tag_key_values
