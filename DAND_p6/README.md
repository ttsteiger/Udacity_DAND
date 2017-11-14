# Make Effective Data Visualizations: Earthquakes Around The World

## Summary

In this project I created an animated visualization of all the significant 
earthquakes that happened between 1956 and 2016. For this visualization I used 
the JavaScript library [D3.js](https://d3js.org/). The animation shows the 
earthquakes for each year separately but the included buttons and checkboxes
allow to look at any selection the user desires.

The data used for this project was collected by the [National Earthquake 
Information Center](https://earthquake.usgs.gov/contactus/golden/neic.php) (NEIC)
and is publicly available on their website. I used a preprocessed data set
from [Kaggle](https://www.kaggle.com/usgs/earthquake-database) which only 
contains earthquakes with a magnitude higher than 5.5 on the Richter scale.





## Design

chart type, visual encodings, and layout


## Feedback

After creating an initial working prototype I discussed the visualization with
friends and co-workers. Many fine iterations happened based on their thoughts
and I want to mention some of them here in this section.

[Prototype](index2.html)

- adding basic control panel


- hue vs. size


- display magnitude upon hover


- feedback on css sytling and page layout (add footer)




## Resources



## Data

columns:

'Date', 'Time', 'Latitude', 'Longitude', 'Type', 'Depth', 'Depth Error',
'Depth Seismic Stations', 'Magnitude', 'Magnitude Type',
'Magnitude Error', 'Magnitude Seismic Stations', 'Azimuthal Gap',
'Horizontal Distance', 'Horizontal Error', 'Root Mean Square', 'ID',
'Source', 'Location Source', 'Magnitude Source', 'Status'

magnitude range:

5.5 -> 9.1

magnitude categories:

http://www.geo.mtu.edu/UPSeis/magnitude.html

depth range:

-1.1 -> 700

depth categories:

https://earthquake.usgs.gov/learn/topics/determining_depth.php

type:

'Earthquake' 'Nuclear Explosion' 'Explosion' 'Rock Burst'

