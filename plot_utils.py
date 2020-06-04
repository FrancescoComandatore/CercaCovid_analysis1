
# plot utilities
import matplotlib.cm
import pandas as pd
import numpy as np


def RGBToHTMLColor(rgb_tuple):
    """ convert an (R, G, B) tuple to #RRGGBB """
    hexcolor = '#%02x%02x%02x' % (int(
        rgb_tuple[0] * 255), int(rgb_tuple[1] * 255), int(rgb_tuple[2] * 255))
    # that's it! '%02x' means zero-padded, 2-digit hex values
    return hexcolor

def HTMLColorToRGB(colorstring):
    """ convert #RRGGBB to an (R, G, B) tuple """
    colorstring = colorstring.strip()
    if colorstring[0] == '#': colorstring = colorstring[1:]
    if len(colorstring) != 6:
        print(ValueError, "input #%s is not in #RRGGBB format" % colorstring)
        return (0, 0, 0)
    r, g, b = colorstring[:2], colorstring[2:4], colorstring[4:]
    r, g, b = [int(n, 16) for n in (r, g, b)]
    return (r, g, b)

# default options for Names and Values over regions
default_dictNamesValues = dict(
    textposition = 'middle center',
    color = 'black',
    size = 12
)

# deault options for boundaries lines
default_dictBoundaries = dict(
    color = 'black',
    width = 1 
)

def map_plot_v2(
        # df containing both plot_key and a geometry column (geodata imported with geopandas)
        data_geometry,
        plot_key,                               # key to plot
        name_key,                               # key for labels
        lower_bound = None,                     # lower bound for colormap
        upper_bound = None,                     # upper bound for colormap
        cmap = matplotlib.cm.RdYlGn,            # colormap
        simplify_factor = 100,                  # factor to simplify shapes
        thresh_percentile = 0.05,               # threshold on lower and upper bound values (used only with lower_bound = None, upper_bound=None)
        width = 600,                            # image size
        height = 600,                           # image size
        plot_showRegionLegend = False,          # plot property: show legend containing the names of the regions and the associated value
        plot_showLegend = False,                # plot property: show a legend containig the dicrete colorbar names
        plot_showColorbar = False,              # plot property: show the colorbar
        plot_showNames = False,                 # plot property: show the name of the shape in the centroid
        plot_showValues = False,                # plot property: show the value of the shape in the centroid
        plot_keyLegendNames = None,                      # plot input: names to show in RegionLegend and Legend
        plot_keyNamesNames = None,                       # plot input: names to show over regions
        plot_keyValuesNames = None,                      # plot input: values to show over regions
        plot_dictNamesValues = default_dictNamesValues,  # plot options: custom options for Names and Values over regions
        plot_dictBoundaries = default_dictBoundaries,    # plot options: custom options for boundaries lines
        plot_colorbarTickvals = [0, 1, 2],               # quando livello e' 'comune' usa [1,4,7]
        plot_colorbarForceIntValues = False,             # plot options: colorbar ticks are integer numbers
        plot_colorbarAddPlusMaxValue = False,            # plot options: add a '+' in upper tick of colorbar
        plot_colorbarAddMinusMinValue = False,           # plot options: add a '-' in lower tick of colorbar
        plot_colorbarAddTextMaxValue = None,             # plot options: text to add after upper tick of colorbar
        plot_colorbarAddTextMinValue = None,             # plot options: text to add after lower tick of colorbar
        plot_title = None
    ):

    ############ Data preparation ################ 
    
    # simplify factor 
    SIMPLIFY_FACTOR = simplify_factor
    THRESH_PERCENTILE = thresh_percentile

    plot_data = []
    data_geometry = data_geometry.sort_values(plot_key)
    
    # colors
    values = data_geometry[plot_key]
    
    if lower_bound is None:
        lower_bound = values.quantile(THRESH_PERCENTILE)
    if upper_bound is None:
        upper_bound = values.quantile(1 - THRESH_PERCENTILE)

    colors = ((values - lower_bound) / (upper_bound - lower_bound))
       
    colors = colors.apply(
        lambda v: None if pd.isnull(v) else RGBToHTMLColor(cmap(v)))

    
    ############ Customize setting ################ 
    
    if plot_showLegend + plot_showRegionLegend + plot_showColorbar > 1:
        print('Error! Choose only one between plot_showLegend, plot_showRegionLegend and plot_showColorbar')
        return dict()
        
    if plot_showColorbar:    
        #colormap_colors = colors
        #if len(colors) >= 20:
        colormap_colors = np.arange(0,1,1./19)
        colormap_colors = [None if pd.isnull(v) else RGBToHTMLColor(cmap(v)) for v in colormap_colors]
              
        rgb_strings = [HTMLColorToRGB(x) for x in colormap_colors]
        colorscale = [[float(idx)/(len(rgb_strings)-1), 
                       "rgb" + str(rgb_strings[idx])] for idx in np.arange(0, len(rgb_strings))]
        
        # where to put tick vals
        colorbar_tickvals = plot_colorbarTickvals
        if len(colorbar_tickvals) < 1:
            colorbar_tickvals = [0, 1, 2]
        
    # Boundaries lines ----------------------------------------------
    dictBoundaries = plot_dictBoundaries
    
    # Legend --------------------------------------------------------
    plot_showlegend = plot_showRegionLegend or plot_showLegend
    if plot_keyLegendNames == None:
        plot_keyLegend = name_key 
    else:
        plot_keyLegend = plot_keyLegendNames
        
    if plot_showLegend:
        plot_legendValues = list(data_geometry[plot_keyLegend].unique())
        data_geometry['needLegendWrite'] = False
        for val in plot_legendValues:
            if len(data_geometry.loc[data_geometry[plot_keyLegend] == val]) > 0:
                idx = data_geometry.loc[data_geometry[plot_keyLegend] == val].iloc[[0]].index
                data_geometry.loc[idx, 'needLegendWrite'] = True        
    
    # Names and Values over regions ---------------------------------
    fontNamesValues = plot_dictNamesValues
    if plot_keyNamesNames == None:
        namesNames = name_key
    else:
        namesNames = plot_keyNamesNames
    if plot_keyValuesNames == None:
        namesValues = plot_key
    else:
        namesValues = plot_keyValuesNames
    text_position = 'middle center'
    if 'textposition' in fontNamesValues.keys():
        text_position = fontNamesValues['textposition']
        del fontNamesValues['textposition']
       
    
    ############# plot ############################
    for index, row in data_geometry.iterrows():
        color = colors[index]
        value = values[index]
        poly_list = []

        if data_geometry['geometry'][index].type == 'Polygon':
            poly_list.append(row.geometry)

        elif data_geometry['geometry'][index].type == 'MultiPolygon':
            for poly in data_geometry['geometry'][index]:
                poly_list.append(poly)
        else:
            print('stop')

        i = 0
        for poly in poly_list:
            #print(row[plot_keyLegend], poly.bounds)
            i = i+1
            
            x = poly.simplify(SIMPLIFY_FACTOR).exterior.xy[0].tolist()
            y = poly.simplify(SIMPLIFY_FACTOR).exterior.xy[1].tolist()
            c_x = poly.simplify(SIMPLIFY_FACTOR).centroid.xy[0].tolist()
            c_y = poly.simplify(SIMPLIFY_FACTOR).centroid.xy[1].tolist()

            ######################### shape #################            
            country_outline = dict(
                type = 'scatter',
                line = dictBoundaries,
                marker = dict(size=0.1),
                x = x,
                y = y,
                fill = 'toself',
                fillcolor = color)
                
            # legend ---------------------
            country_outline['showlegend'] = plot_showlegend
            country_outline['legendgroup'] = "shapes"
            
            if plot_showRegionLegend:
                country_outline['name'] = row[plot_keyLegend] + " (%.2f)" % value # region + value 
            elif plot_showLegend:
                if row['needLegendWrite']:
                    country_outline['name'] = row[plot_keyLegend]
                else:
                    country_outline['showlegend'] = False # to avoid duplicates
                
            if i > 1 and plot_showlegend:
                country_outline['showlegend'] = False # to avoid duplicates
  
            # hover info ------------------
            country_outline['hoverinfo']='none'
            
            ######################### hover #################
                
            hover_point = dict(
                type = 'scatter',
                showlegend = False,
                legendgroup = "centroids",
                name = row[name_key],
                marker = dict(size=0.1, color=color),
                x = c_x,
                y = c_y,
                text = "%.2f" % value,
                hoverinfo = "text+name")
            
            if plot_showNames or plot_showValues:
                if i == 1:
                    string = ""
                    if plot_showNames:
                        string = string + row[namesNames]
                    if plot_showNames and plot_showValues:
                        string = string + "<br>"    # equivalent of \n
                    if plot_showValues:
                        curr_valueName = row[namesValues]
                        if type(curr_valueName) != str:
                            string = string + "%.2f" % curr_valueName
                        else:
                            string = string + curr_valueName

                    hover_point['mode'] = 'text'
                    hover_point['text'] = string
                    hover_point['textfont'] = fontNamesValues
                    hover_point['textposition'] = text_position
            
            plot_data.append(country_outline)
            plot_data.append(hover_point)
            
            
    ############ colorbar ################ 
    if plot_colorbarForceIntValues:
        ticktext = ["%d" % int(lower_bound), 
                    "%d" % int(float((upper_bound - lower_bound)/2)), 
                    "%d" % int(upper_bound)]
    else:
        ticktext = ["%.2f" % lower_bound, 
                    "%.2f" % float(float((upper_bound - lower_bound)/2)), 
                    "%.2f" % upper_bound]
        
    if plot_colorbarAddPlusMaxValue:
        ticktext[-1] = ticktext[-1] + '+'
    if plot_colorbarAddTextMaxValue != None:
        ticktext[-1] = ticktext[-1] + plot_colorbarAddTextMaxValue
    if plot_colorbarAddMinusMinValue:
        ticktext[0] = ticktext[0] + '-'
    if plot_colorbarAddTextMinValue != None:
        ticktext[0] = ticktext[0] + plot_colorbarAddTextMinValue
        
    if plot_showColorbar:
        colormap_line = dict(
            type='scatter',
            showlegend=False,
            x = [c_x[0] for x in colorscale],
            y = [c_y[0] for x in colorscale],
            marker=dict(
                size=0.1,
                colorscale = colorscale,
                colorbar = dict(
                    #title = 'Surface Heat',
                    #titleside = 'top',
                    tickmode = 'array',
                    tickvals = colorbar_tickvals,
                    #ticktext = ["%.2f" % values.iloc[0], 
                    #            "%.2f" % float(float((values.iloc[-1] - values.iloc[0])/2)), 
                    #            "%.2f" % values.iloc[-1]],
                    ticktext = ticktext,
                    #ticktext = ['Hot','Mild','Cool'],
                    ticks = 'outside'
                )
            ),
            hoverinfo = 'none'
        )
        plot_data.append(colormap_line)

    #print('--------------')  
    #print(colormap_line)
    ############ Layout ################    
    layout = dict(
        hovermode='closest',
        xaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            scaleanchor= "x",
            autorange=True,
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        margin=dict(t=40, b=40, r=40, l=40),
        width=width,
        height=height,
    )
    if (plot_title):
        layout['title'] = plot_title
    
        
    return dict(data=plot_data, layout=layout)
    
def append_prov_borders(fig, plot_data_prov, 
                        plot_key = 'value', 
                        shp_name_key = 'DEN_CMPRO', 
                        plot_dictBoundaries = None,
                        plot_showNames = False,
                        plot_keyNamesNames = None,
                        plot_dictNamesValues = default_dictNamesValues
):
    if not plot_dictBoundaries:
        plot_dictBoundaries = dict(color = 'black', width = 3)   

    fig_prov = map_plot_v2(plot_data_prov, 
                      plot_key, 
                      shp_name_key,
                      cmap = matplotlib.cm.RdYlGn_r,
                      plot_dictBoundaries = plot_dictBoundaries,
                      plot_showNames = plot_showNames,
                      plot_keyNamesNames = plot_keyNamesNames,
                      plot_dictNamesValues = plot_dictNamesValues
                    )
    # transparency 100%
    for d in fig_prov['data']:
        d['fillcolor'] = "rgba(0,0,0,0)"
        fig['data'].append(d)
        
    return fig