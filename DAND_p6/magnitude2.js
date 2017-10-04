// magnitude.js

"use strict";

function draw(geo_data) {

    //graphic settings
    var margin = 75,
        width = 1920 - margin,
        height = 1080 - margin,
        rad_min = 5,
        rad_max = 20;

    // list containing all years
    var years = [];
    for (var i = 1965; i <= 2016; i += 1) {
        years.push(i);
      }

    var svg = d3.select("body")
        .append("svg")
          .attr("width", width + margin)
          .attr("height", height + margin);

    var projection = d3.geo.mercator()
                           .scale(220)
                           .translate([width / 2, height / 1.5]);

    var path = d3.geo.path().projection(projection);

    var map = svg.append('g')
                 .attr('class', 'map')
                 .selectAll('path')
                 .data(geo_data.features)
                 .enter()
                 .append('path')
                 .attr('d', path)
                 .style('fill', 'white')
                 .style('stroke', 'black')
                 .style('stroke-width', 0.5);
  
    // plot points for all earthquakes, radius based on strength
    function plot_mag_points(data) {
        
        var magnitude_extent = d3.extent(data, function(d) {
            return d['Magnitude'];
        });

        var radius= d3.scale.sqrt()
                            .domain(magnitude_extent)
                            .range([rad_min, rad_max]);
        
        // group data by years
        var nested_data = d3.nest()
                            .key(function(d) {
                                return d['Date'].getUTCFullYear();
                            })
                            .entries(data)

        // add bubble for each earthquakt to svg map, coordinates are 
        // converted using the Mercator projection and the radius is 
        // calculated from the magnitude value
        
        // for the display of all the data in the nested structure we start
        // by creating group elements for each year group
        var year_groups = svg.selectAll(".bubbles")
                             .data(nested_data)
                             .enter().append("g")
                             .attr("class", "bubbles")
                             .attr("id", function(d) {
                                 return "y" + d['key'];
                             });
        
        // create svg circles for all earthquakes within the different 
        // year groups
        var bubbles = year_groups.selectAll("circle")
                            .data(function(d) {
                                return d.values;
                            })
                            .enter()
                            .append("circle")
                            .attr("cx", function(d) {
                               return projection([+d['Longitude'], 
                                                  +d['Latitude']])[0];
                            })
                            .attr("cy", function(d) {
                                return projection([+d['Longitude'], 
                                                   +d['Latitude']])[1];
                            })
                            .attr('r', function(d) {
                                return radius(d['Magnitude'])
                            });

        // create legend
        var legend = svg.append('g')
          .attr("class", "legend")
          .attr("transform", "translate(" + (width - 100) + "," + 20 + ")")
          .attr("height", 200)
          .attr("width", 200)
          .selectAll("g")
          .data(["6", "7", "8", "9"])
          .enter();

        // legend title
        legend.append("text")
          .text("Magnitude")

        // create bubbles for legend 
        legend.append("circle")
          .attr("cx", 20)
          .attr("cy", function(d, i) {
              return i * 50 + 30;
          })
          .attr("r", function(d) {
              return radius(+d)

          });

        // label the legend bubbles
        legend.append("text")
          .attr("y", function(d, i) {
            return i * 50 + 35;
          })
          .attr("x", 10 * 5)
          .text(function(d) {
            return d;
          });

        function update_mag_year(year) {
            // display only the earthquakes from a certain year
            
            // only select data for the specified year
            // var filtered_data = nested_data.filter(function(d) {
            //    return new Date(d['key']).getUTCFullYear() === year;
            //});

            
            // adjust title
            d3.select('h2')
              .text("Earthquakes " + year);

            // select all bubbles that correspond with the specified year
            var bubbles_year = svg.selectAll("#y" + year + " circle");

            // display bubbles from current year
            bubbles_year.transition()
                        .duration(100)
                        .style('opacity', 0.7);
        }

        // year animation
        var year_idx = 0;

        var year_interval = setInterval(function() {
            if (year_idx === 0) {
              // hide all bubbles in the first animation frame 
              svg.selectAll(".bubbles circle")
                 .transition()
                 .duration(0)
                 .style('opacity', 0.0);
            }

            // add bubbles for current year
            update_mag_year(years[year_idx]);
            year_idx++;

            // finish animation loop when last year is reached
            if (year_idx >= years.length) {
                clearInterval(year_interval);
            }
        }, 600);
    }

    // format used to parse date from string in .csv
    var format = d3.time.format("%m/%d/%Y");

    // load .csv file and rund plot_mag_points function
    d3.csv('earthquakes_edited.csv', function(d) {
            d['Magnitude'] = parseFloat(d['Magnitude']) // convert to float
            d['Date'] = format.parse(d['Date']) // parse date from string
            return d;
        }, plot_mag_points);
}