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

Key messages from the plot...

Distribution along tectonic plate boundaries

Types of boundaries visible as strong earthquakes mainly happen in specific places

Evolution over time?


## Design

The final visualization I ended up with is a bubble chart on top of a mercator
projection of the globe. The magnitude is encoded in the bubble size as well as
the color of the bubbles. As the Richter scale is a logarithmic scale, meaning
that a magnitude increase of 1 denotes an earthquake that is 10 times stronger, 
I think that using two visual encodings simultaneously is a reasonable measure
to reinforce the differences. Including color also allows to gain better
information when all earthquakes are displayed simultaneously and the data points
overlie with each other.

For the bubble size also logarithmic scale


red for severe earthquakes and green for weaker ones as most people associate
danger with the color red




chart type, visual encodings, and layout


## Feedback

After creating an initial working prototype I discussed the 
visualization with friends and co-workers. Many fine iterations happened based 
on their thoughts and I want to mention some of them here in this section.

### First Iteration

Starting from a simple animation that plays upon page load and displays all the 
earthquakes accumulated over the years I started working on a control panel to 
add some interactivity to the page. 

![Prototype](image1.png)

My intent was to let the user choose if he wants to look at the data from 
individual years, a custom selection or just observe a time lapse over the whole
timespan. The picture below shows the final button and checkbox configuration I
ended up with.

![Control Panel](image2.png)


- adding basic control panel

### Second Iteration

- hue vs. size and change of animation from overlaying to displaying only a single year

### Third Iteration

- display magnitude and additional information upon hover






## Resources

* D3.js
* NEIC
* chromatic color scale



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

