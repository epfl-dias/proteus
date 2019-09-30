import Component from '@ember/component';
import { tree, hierarchy } from 'd3-hierarchy';
/* eslint-disable no-unused-vars */
import { transition } from 'd3-transition';
/* eslint-enable no-unused-vars */
import { zoom , zoomTransform } from 'd3-zoom';
// import { transform, linkHorizontal, linkVertical } from 'd3-shape';
// import { transform } from 'd3-shape';
import { select, event } from 'd3-selection';
// import { drag } from 'd3-drag';

import $ from 'jquery';

export default Component.extend({
    tagName: 'svg',
    data: null,
    // totalNodes: 0,
    // maxLabelLength: 0,
    // // variables for drag/drop
    // selectedNode: null,
    // draggingNode: null,
    // // panning variables
    // panSpeed: 200,
    // panBoundary: 20, // Within 20px from edges will pan when dragging.
    // // Misc. variables
    // i: 0,
    duration: 500,
    // root: null,
    vertical: true,
    nodeRad: 20,
    // viewerWidth: 500,
    // viewerHeight: 500function(treeData) {
    classNames: ['container'],
    // treeData: null,
    fix_expression_format(e){
        if (e instanceof Array){
            return e.map(x => this.fix_expression_format(x));
        }
        if (typeof e === 'string' || e instanceof String){
            return e;
        }
        for(var index in e) { 
           if (e.hasOwnProperty(index)) {
               e[index] = this.fix_expression_format(e[index]);
           }
        }
        var s;
        if (typeof e === 'object' && 'expression' in e){
            switch (e.expression){
                case "bool":
                    if (e.v){
                        s = "true";
                        break;
                    } else {
                        s = "false";
                        break;
                    }
                case "dstring":
                case "string":
                    s = "'" + e.v + "'";
                    break;
                case "int":
                    s = "" + e.v;
                    break;
                case "int64":
                    s = "" + e.v + "ull";
                    break;
                case "add":
                    s = "(" + e.left + " + " + e.right + ")";
                    break;
                case "div":
                    s = "(" + e.left + " / " + e.right + ")";
                    break;
                case "multiply":
                    s = "(" + e.left + " * " + e.right + ")";
                    break;
                case "eq":
                    s = "(" + e.left + " = " + e.right + ")";
                    break;
                case "le":
                    s = "(" + e.left + " <= " + e.right + ")";
                    break;
                case "lt":
                    s = "(" + e.left + " < " + e.right + ")";
                    break;
                case "gt":
                    s = "(" + e.left + " > " + e.right + ")";
                    break;
                case "ge":
                    s = "(" + e.left + " >= " + e.right + ")";
                    break;
                case "or":
                    s = "(" + e.left + " or " + e.right + ")";
                    break;
                case "and":
                    s = "(" + e.left + " and " + e.right + ")";
                    break;
                case "sub":
                    s = "(" + e.left + " - " + e.right + ")";
                    break;
                case "neg":
                    s = "(-)" + e.e;
                    break;
                case "argument":
                    s = e.type.relName;
                    break;
                case "recordProjection":
                    s = e.attribute.relName + "." + e.attribute.attrName;
                    break;
                case "recordConstruction":
                    s = "{";
                    var p = []
                    for (var t in e.attributes){
                        p.push(this.fix_expression_format(e.attributes[t].e) + " AS " + e.attributes[t].name)
                    }
                    s = s + p.join(", ") + "}";
                    break;
                default:
                    console.log(e)
                    s = e;
                    break;
            }
            if ("register_as" in e){
                if ("isBlock" in e["register_as"] && e["register_as"]["isBlock"]){
                    s = "[" + s + "]";
                }
                s = s + " AS " + e.register_as.relName + "." + e.register_as.attrName;
            }
            return s;
        }
        return e;
    },
    fix_format(d){
        var children = [];
        var name     = "";
        var x            = {};

        x.gpu = false;
        if (d.gpu) x.gpu = d.gpu;
        x.blockwise = false;
        if (d.blockwise) x.blockwise = d.blockwise;
        if (d.operator == "gpu-to-cpu") x.gpu2cpu = true;
        if (d.operator == "cpu-to-gpu") x.cpu2gpu = true;
        // d.is_plugin = false;
        if (d.operator){
            name     = d.operator;
        // } else if (d.type){
            // d.is_plugin = true;
            // name     = "plugin-" + d.type
        }
        if (d.input != null){
            if (d.input instanceof Array){
                d.input.forEach(function(c){children.push(c);});
            } else {
                children.push(d.input);
            }
            // delete d.input;
        }
        if (d.build_input != null){
            children.push(d.build_input);
            // delete d.build_input;
        }
        if (d.probe_input != null){
            children.push(d.probe_input);
            // delete d.probe_input;
        }
        if (d.plugin != null){
            name    += " (" + d.plugin.type + ")";
            x.attrs  = d.plugin.name;
            // children.push(d.plugin);
            // delete d.plugin;
            if (d.plugin.input != null){
                children.push(d.plugin.input);
                delete d.plugin.input;
            }
        }
        x.value      = $.extend(true, {}, d);
        for(var index in d) {
           if (d.hasOwnProperty(index)) {
                x.value[index] = this.fix_expression_format(d[index])
           }
        }

        delete x.value.input;
        delete x.value.build_input;
        delete x.value.probe_input;

        if (x.value.hasOwnProperty("plugin")){
            delete x.value.plugin.input;
        }
        
        delete x.value.operator     ;
        delete x.value.type         ;
        // delete x.value.projections  ;
        delete x.value.name         ;
        delete x.value.output       ;
        delete x.value.gpu          ;
        // delete x.value.blockwise    ;
        // delete x.value.jumpTo       ;
        x.name       = name;
        x.children   = children.map(c => this.fix_format(c));
        return x;
    },
    init(){
        this._super(...arguments);

        this.set('treeData', this.fix_format(this.get('data')));
    },
    didInsertElement(){
        this._super(...arguments);

        var treeData = this.get('treeData');
        var nodeRad  = this.get('nodeRad' );

        // Set the dimensions and margins of the diagram
        var margin = {top: 0, right: 0, bottom: 0, left: 0},
            width = 1920 - margin.left - margin.right,
            height = 400 - margin.top - margin.bottom;

        const vertical = this.get('vertical');

        // append the svg object to the body of the page
        // appends a 'group' element to 'svg'
        // moves the 'group' element to the top left margin
        var svg = select(this.$().get(0))
                .attr("width", width + margin.right + margin.left)
                .attr("height", height + margin.top + margin.bottom)
                .attr("display", "block")
                .attr("class", "overlay")
                // .call(zoomListener)
            .append("g")
                .attr("transform", "translate("
                    + margin.left + "," + margin.top + ")");

        // define the zoomListener which calls the zoom function on the "zoom" event constrained within the scaleExtents
        var zoomListener = zoom().scaleExtent([0.1, 3]).on("zoom", function() {
            svg.attr("transform", event.transform);//"translate(" + event.translate + ")scale(" + event.scale + ")");
        }, true).touchable(true);

        select(this.$().get(0)).call(zoomListener)
        
        var div = select("body").append("div")
            .attr("class", "ttip")
            // .attr("stype", "padding-top:0px;padding-right:0px;padding-bottom:0px;padding-left:0px")
            .style("opacity", 0);

        const this_ = this;
        // window.onresize = function(/*event*/) {
        //     var viewerWidth  = this_.$(document).width();
        //     var viewerHeight = this_.$(document).height();
            
        //     var svg    = select(this_.$().get(0));
        //     var offset = this_.$().offset();

        //     svg.attr("width", viewerWidth - offset.left + (margin.right + margin.left))
        //             .attr("height", viewerHeight - offset.top + (margin.top + margin.bottom));

        //     update(root);
        // };
        // window.onresize(); //let it get span the window
        
        var viewerWidth  = select(this.$().get(0)).attr('width' );
        // var viewerHeight = select(this.$().get(0)).attr('height');
        
        var i = 0,
            root;

        // declares a tree layout and assigns the size
        // var treemap = tree().size([height, width]);
        var treemap = tree();

        // Assigns parent, children, height, depth
        root = hierarchy(treeData, function(d) { return d.children; });
        root.x0 = viewerWidth / 2;
        root.y0 = 0;

        // Collapse after the second level
        // root.children.forEach(collapse);

        // update(root, true);

        // Collapse the node and all it's children
        // function collapse(d) {
        //   if(d.children) {
        //     d._children = d.children
        //     d._children.forEach(collapse)
        //     d.children = null
        //   }
        // }

        var duration = this.get('duration');
        
        function update(source, rescaleToFit) {
            // Assigns the x and y position for the nodes
            var transform    = zoomTransform(select(this_.$().get(0)).node());
            var k            = (rescaleToFit) ? transform.k : 1;
            var viewerWidth  = select(this_.$().get(0)).attr('width' )*0.9/k;
            var viewerHeight = select(this_.$().get(0)).attr('height')*0.9/k;

            // viewerWidth      = transform.invertX(viewerWidth );
            // viewerHeight     = transform.invertX(viewerHeight);
            
            if (vertical)   treemap.size([viewerWidth, viewerHeight]);
            else            treemap.size([viewerHeight, viewerWidth]);

            var treeData = treemap(root);

            // Compute the new tree layout.
            var nodes = treeData.descendants(),
              links = treeData.descendants().slice(1);

            // Normalize for fixed-depth.
            nodes.forEach(function(d){ d.y = nodeRad * 2 + d.depth * ((vertical) ? 90 : 180)});

            // ****************** Nodes section ***************************

            // Update the nodes...
            var node = svg.selectAll('g.node')
              .data(nodes, function(d) {return d.id || (d.id = ++i); });

            // Enter any new modes at the parent's previous position.
            var nodeEnter = node.enter().append('g')
                .attr('class', 'node')
                .attr("transform", function(/*d*/) {
                    if (vertical) return "translate(" + source.x0 + "," + source.y0 + ")";
                    return "translate(" + source.y0 + "," + source.x0 + ")";
                })
                .on('click', click)
                ;

            // Add Circle for the nodes
            nodeEnter.append('circle')
                .attr('class', 'node')
                .attr('r', 1e-6)
                .on("mouseover", function(v) {
                    var d = v.data;
                    if (!($.isEmptyObject(d.value))) {
                      // d3.select(this).select("text").style("visibility", "hidden") 
                        var s = JSON.stringify(d.value, null, 2)
                        console.log(s)
                        div.transition()        
                            .duration(200)      
                            .style("opacity", .9);      
                        div.html("<pre>" + s + "</pre>")
                            .style("left", (event.pageX) + "px")     
                            .style("top", (event.pageY - 28) + "px");    
                    }
                })
                // .on("mouseout", function(d) { 
                //   d3.select(this).select("text").style("visibility", "visible") 
                // })
                .on("mouseout", function(v) {
                    var d = v.data;
                    if (!($.isEmptyObject(d.value))) {
                        div.transition()        
                            .duration(500)      
                            .style("opacity", 0);
                        div.html("")
                            .style("left", 0 + "px")
                            .style("top", 0 + "px");
                    }
                });

            // Add labels for the nodes
            nodeEnter.append('text')
                .attr("dy", function(/*d*/) {
                    if (vertical) return "0.33em";
                    return (0.33 + 2) + "em";
                })
                .attr("x", function(/*d*/) {
                    if (vertical) return -25;//d.children || d._children ? -25 : 25;
                    return nodeRad;
                })
                // .attr("dx", ".35em")
                // .attr("y", function(d) {
                //     return d.children || d._children ? -25 : 25;
                // })
                .attr("text-anchor", function(/*d*/) {
                    if (vertical) return "end";//d.children || d._children ? "end" : "start";
                    return "end";
                })
                .attr("pointer-events", "none")
                .text(function(d) { return d.data.name; });

            // UPDATE
            var nodeUpdate = nodeEnter.merge(node);

            // Transition to the proper position for the node
            nodeUpdate.transition()
                .duration(duration)
                .attr("transform", function(d) { 
                    if (vertical) return "translate(" + d.x + "," + d.y + ")";
                    return "translate(" + d.y + "," + d.x + ")";
                });

            // Update the node attributes and style
            nodeUpdate.select('circle.node')
                // .attr('r', 10)
                // .style("fill", function(d) {
                //     return d._children ? "lightsteelblue" : "#fff";
                // })
                .attr('cursor', 'pointer')
                .attr("r", nodeRad)
                .attr('class', function(v){
                    var d = v.data;
                    if ( d.cpu2gpu &&  d.blockwise) return "cpu2gpuBlockNode";
                    if ( d.cpu2gpu && !d.blockwise) return "cpu2gpuNode"     ;
                    if ( d.gpu2cpu &&  d.blockwise) return "gpu2cpuBlockNode";
                    if ( d.gpu2cpu && !d.blockwise) return "gpu2cpuNode"     ;
                    if ( d.gpu     &&  d.blockwise) return "gpuBlockNode"    ;
                    if ( d.gpu     && !d.blockwise) return "gpuNode"         ;
                    if (!d.gpu     &&  d.blockwise) return "cpuBlockNode"    ;
                    if (!d.gpu     && !d.blockwise) return "cpuNode"         ;
                })
                // .style("stroke", function(d){
                //     return d.gpu ? "green" : "steelblue";
                // })
                // .style("stroke-width", function(d){
                //     return d.gpu ? "2.5px" : "1.5px";
                // })
                // .style("fill", function(d) {
                //     return d._children ? "lightsteelblue" : "#fff";
                // })
                ;

            // Remove any exiting nodes
            var nodeExit = node.exit().transition()
                .duration(duration)
                .attr("transform", function(/*d*/) {
                    if (vertical) return "translate(" + source.x + "," + source.y + ")";
                    return "translate(" + source.y + "," + source.x + ")";
                })
                .remove();

            // On exit reduce the node circles size to 0
            nodeExit.select('circle')
                .attr('r', 1e-6);

            // On exit reduce the opacity of text labels
            nodeExit.select('text')
                .style('fill-opacity', 1e-6);

            // ****************** links section ***************************

            // Update the links...
            var link = svg.selectAll('path.link')
                .data(links, function(d) { return d.id; });

            // Enter any new links at the parent's previous position.
            var linkEnter = link.enter().insert('path', "g")
                .attr("class", "link")
                .attr('d', function(/*d*/){
                    var o = {x: source.x0, y: source.y0}
                    return diagonal(o, o)
                });

            // UPDATE
            var linkUpdate = linkEnter.merge(link);

            // Transition back to the parent element position
            linkUpdate.transition()
                .duration(duration)
                .attr('d', function(d){ return diagonal(d, d.parent) });

            // Remove any exiting links
            // var linkExit = 
            link.exit().transition()
                .duration(duration)
                .attr('d', function(/*d*/) {
                    var o = {x: source.x, y: source.y}
                    return diagonal(o, o)
                })
                .remove();

            // Store the old positions for transition.
            nodes.forEach(function(d){
                d.x0 = d.x;
                d.y0 = d.y;
            });

            // Creates a curved (diagonal) path from parent to the child nodes
            function diagonal(s, d) {
                if (vertical) {
                    return `M ${s.x} ${s.y}
                            C ${s.x} ${(s.y + d.y) / 2} ,
                              ${d.x} ${(s.y + d.y) / 2},
                              ${d.x} ${d.y}`
                } else {
                    return `M ${s.y}             ${s.x}
                            C ${(s.y + d.y) / 2} ${s.x},
                              ${(s.y + d.y) / 2} ${d.x},
                              ${d.y}             ${d.x}`
                }
            }

            // Toggle children on click.
            function click(d) {
                if (d.children) {
                    d._children = d.children;
                    d.children = null;
                } else {
                    d.children = d._children;
                    d._children = null;
                }
                update(d, false);
            }
        }


        window.onresize = function(/*event*/) {
            var viewerWidth  = this_.$(document).width();
            var viewerHeight = this_.$(document).height();
            
            var svg    = select(this_.$().get(0));
            var offset = this_.$().offset();

            svg.attr("width", viewerWidth - offset.left + (margin.right + margin.left))
                    .attr("height", viewerHeight - offset.top + (margin.top + margin.bottom));

            update(root, true);
        };
        window.onresize(); //let it get span the window
    },
    didDestroyElement(){
        window.onresize = function(){};
    }
});
