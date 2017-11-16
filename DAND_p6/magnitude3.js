// magnitude.js

"use strict";

function draw(geo_data) {

    // graphic settings
    var width = 1920,
        height = 1080,
        rad_min = 5,
        rad_max = 20;

    // svg element
    var svg = d3.select(".rightpane")
                .append("svg")
                  .attr("width", width)
                  .attr("height", height)
                  .attr("x", 0)
                  .attr("y", 50);

    // function to convert lon and lat into pixel coordinates
    var projection = d3.geo.mercator()
                           .scale(230)
                           .translate([width / 2, height / 1.6]);
    
    // add world map to svg element
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
          .attr("transform", "translate(" + (width - 200) + "," + 20 + ")")
          .attr("height", 200)
          .attr("width", 200)
          .selectAll("g")
          .data(["6", "7", "8", "9"])
          .enter();

        // legend title
        svg.select(".legend")
           .append("text")
           .text("Magnitude")
           .attr("x", 40)
           .attr("y", -5)
           .style("font-size", "20px");

        // create bubbles for legend 
        legend.append("circle")
          .attr("cx", 60)
          .attr("cy", function(d, i) {
              return i * 40 + 25;
          })
          .attr("r", function(d) {
              return radius(+d)

          });

        // label the legend bubbles
        legend.append("text")
          .attr("y", function(d, i) {
            return i * 40 + 30;
          })
          .attr("x", 90)
          .text(function(d) {
            return d;
          })
          .style("font-size", "20px");
    }  

    // format used to parse date from string in .csv
    var format = d3.time.format("%m/%d/%Y");

    // load .csv file and rund plot_mag_points function
    d3.csv('earthquakes_edited.csv', function(d) {
            d['Magnitude'] = parseFloat(d['Magnitude']) // convert to float
            d['Date'] = format.parse(d['Date']) // parse date from string
            return d;
        }, plot_mag_points);

    // list containing all years
    var years = [];
    for (var i = 1965; i <= 2016; i += 1) {
        years.push(i);
    }

    // add checkboxes for all years
    var checkbox_list = d3.select(".leftpane")
                          .append("div")
                            .attr("class", "selection-display")
                          .append("ul")
                            .attr("class", "checkbox-grid")
                          .selectAll("ul")
                          .data(years)
                          .enter()
                          .append("li");
                          

    var checkboxes = checkbox_list.append("input")
                                  .attr("class", "year-checkbox")
                                  .attr("type", "checkbox")
                                  .attr("name", "year")
                                  .attr("value", function(d) {
                                     return d;
                                  });

    var checkbox_labels = checkbox_list.append("label")
                                         .text(function(d) {
                                            return d;
                                         });

    // add select/unselect all buttons
    var select_all_button = d3.select(".leftpane .selection-display")
                              .append("div")
                                .attr("class", "select-buttons")
                              .append("button")
                                .attr("type", "button")
                                .attr("onclick", "set_all_checkboxes(true)")
                                .text("Select All");

    var unselect_all_button = d3.select(".leftpane .selection-display .select-buttons")
                                .append("button")
                                .attr("type", "button")
                                .attr("onclick", "set_all_checkboxes(false)")
                                .text("Unselect All");

    // add display selection button
    var button = d3.select(".leftpane")
                   .append("div")
                   .append("button")
                     .attr("type", "button")
                     .attr("onclick", "display_selection()")
                     .text("Display Selection");
}

function update_mag_year(year, svg) {
        // display only the earthquakes from a certain year

        // display year
        d3.select(".rightpane h2")
          .text(year);

        // select all bubbles that correspond with the specified year
        var bubbles_year = svg.selectAll("#y" + year + " circle");

        // display bubbles from current year
        bubbles_year.transition()
                    .duration(100)
                    .style('opacity', 0.4);
}

function run_animation() {

    // select svg element
    var svg = d3.select(".rightpane svg")

    // list containing all years
    var years = [];
    for (var i = 1965; i <= 2016; i += 1) {
        years.push(i);
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
        update_mag_year(years[year_idx], svg);
        year_idx++;

        // finish animation loop when last year is reached
        if (year_idx >= years.length) {
            clearInterval(year_interval)
        }
    }, 600);
}

function set_all_checkboxes(check_value) {
    //

    var checkboxes = document.getElementsByName('year');

    for (var i = 0; i < checkboxes.length; i++) { 
       if (check_value === true) {
          checkboxes[i].checked = true;
       } else {
          checkboxes[i].checked = false;
       }
    }
}

function display_selection() {

    // select svg element
    var svg = d3.select(".rightpane svg")

    // hide all bubbles in the first animation frame 
    svg.selectAll(".bubbles circle")
       .transition()
       .duration(0)
       .style('opacity', 0.0);

    // find all the checked boxes and the correpsonding values 
    var checked_boxes = document.querySelectorAll('.year-checkbox:checked');

    // loop trough all check box values and add the corresponding points to the
    // map
    for (var i = 0; i < checked_boxes.length; i++) {
      var year = checked_boxes[i].value;

      update_mag_year(year, svg);
    }

    // set title
    d3.select(".rightpane h2")
        .text("Selection");


    


}