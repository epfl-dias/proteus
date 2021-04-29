import Component from '@ember/component';
import {brushX} from 'd3-brush';
import {scaleBand, scaleLinear, scaleOrdinal} from 'd3-scale'; //scalePoint
import {
  scaleImplicit,
  schemeDark2
} from 'd3-scale-chromatic'; //schemeDark2, schemeAccent
import {event, select} from 'd3-selection';
import {zoom, zoomIdentity, zoomTransform} from 'd3-zoom';

// Based on:
// https://medium.freecodecamp.org/d3-and-canvas-in-3-steps-8505c8b27444
export default Component.extend({
  data : null,
  custom : null,
  whiteSpace : 0.01, // fraction of space reserver for whitespace between lines
  height : 512,
  width : 512,
  mapHeightPercent : 0.2,
  mapHeight : 0.2,
  mainCanvas : null,
  map : null,
  mapImage : null,
  xScale : null,
  xScale_brush : null,
  yScale : null,
  zoomListener : null,
  didInsertElement() {
    this.get('data').forEach((d, index) => {
      let ind = index + 1; // 0 is used for empty space
      var ret = [ ind & 0xff, (ind & 0xff00) >> 8, (ind & 0xff0000) >> 16 ];
      var col = "rgb(" + ret.join(',') + ")";
      d.rgbid = col;
    });

    let top = select("#container").node().getBoundingClientRect().top;
    let mapHeight =
        (this.$(document).height() - top) * this.get('mapHeightPercent');
    this.set('mapHeight', mapHeight);

    this.set('height', this.$(document).height() - top - mapHeight);
    this.set('width', this.$(document).width());

    let svgHeight = this.get('height');
    let svgWidth = this.get('width');

    let mainCanvas =
        select('#container')
            .append('canvas')
            .classed('mainCanvas', true) // select(this.$('canvas')[0])
            .attr("top", 0)
            .attr("left", 0)
            .attr('style', 'line-height:0')
            .attr("touch-action", "none")
            .attr("position", "absolute")
            .attr('height', svgHeight)
            .attr('width', svgWidth);

    let hiddenCanvas =
        select('#container')
            .append('canvas')
            .classed('hiddenCanvas', true) // select(this.$('canvas')[1])
            .attr("top", 0)
            .attr("left", 0)
            .attr('style', 'line-height:0')
            .attr("position", "absolute")
            .attr('height', svgHeight)
            .attr('width', svgWidth);

    let this_ = this;
    mainCanvas.on('mousemove', function() {
      this_.draw(hiddenCanvas, true); // Draw the hidden canvas.

      // Get mouse positions from the main canvas.
      var mouseX = event.layerX || event.offsetX;
      var mouseY = event.layerY || event.offsetY;

      var hiddenCtx = hiddenCanvas.node().getContext('2d');

      // Pick the colour from the mouse position.
      var col = hiddenCtx.getImageData(mouseX, mouseY, 1, 1).data;
      // Then stringify the values in a way our map-object can read it.
      var colKey = col[0] | (col[1] << 8) | (col[2] << 16);

      // Get the data from our map!
      if (colKey != 0) {
        var nodeData = this_.get('data')[colKey - 1];

        select('#tooltip')
            .style('opacity', 0.8)
            .style('top', event.pageY + 5 + 'px')
            .style('left', event.pageX + 5 + 'px')
            .style('height', 0)
            .style('width', 0)
            .html(JSON.stringify(nodeData, null, 2));
      } else {
        select('#tooltip')
            .style('opacity', 0)
            .style('height', 0)
            .style('width', 0)
            .html("");
      }
    });

    var zoomListener =
        zoom()
            .translateExtent([ [ 0, 0 ], [ svgWidth, 0 ] ])
            .scaleExtent([ 1, Infinity ])
            .on("zoom",
                function() {
                  if (event.sourceEvent && event.sourceEvent.type === "brush")
                    return; // ignore zoom-by-brush
                  var t = event.transform;
                  // var xScale_brush = this_.get('xScale_brush');
                  var xScale = this_.get('xScale');
                  // xScale.domain(t.rescaleX(xScale_brush).domain());
                  this_.get('map').call(this_.get('brushListener').move,
                                        xScale.range().map(t.invertX, t));
                  this_.draw(mainCanvas, false);
                  this_.draw(hiddenCanvas, true);
                },
                true)
            .touchable(true);

    // this.set('zoomer', zoomListener);
    this.set('zoomListener', zoomListener);
    mainCanvas.call(zoomListener);

    this.set('mainCanvas', mainCanvas);
    this.databind();
    console.log(select("#container").node().getBoundingClientRect().top);

    window.onresize = function(/*event*/) {
      // this_.get('brushListener')
      // console.log("asdasd");
      // var extend = this_.get('brushListener').extend();
      // console.log("asdasd");
      // let old_mapHeight = this_.get('mapHeight');
      let top = select("#container").node().getBoundingClientRect().top;
      let mapHeight =
          (this_.$(document).height() - top) * this_.get('mapHeightPercent');
      this_.set('mapHeight', mapHeight);

      let old_svgHeight = this_.get('height');
      let old_svgWidth = this_.get('width');

      this_.set('height', this_.$(document).height() - top - mapHeight);
      this_.set('width', this_.$(document).width());

      let svgHeight = this_.get('height');
      let svgWidth = this_.get('width');

      this_.get('yScale').range([ 0, svgHeight ]);
      this_.get('xScale').range([ 0, svgWidth ]);

      mainCanvas.attr('width', svgWidth).attr('height', svgHeight);

      hiddenCanvas.attr('width', svgWidth).attr('height', svgHeight);

      this_.get('map').attr('width', svgWidth).attr('height', mapHeight);
      this_.get('mapImage').attr('width', svgWidth).attr('height', mapHeight);

      this_.draw(mainCanvas, false);

      if (svgHeight > old_svgHeight || svgWidth > old_svgWidth) {
        // quality has been improved in at least one dimension
        // update map
        this_.get('mapImage').attr('href', mainCanvas.node().toDataURL());
      }

      this_.get('brushListener').extent([
        [ 0, 0 ], [ svgWidth, this_.get('mapHeight') ]
      ])

      // // // var s = [this_.get('brushListener').extend()[0][0],
      // extend()[1][0]];
      // // // console.log(s);
      // //         // xScale.domain(s.map(xScale_brush.invert, xScale_brush))
      // //       // if (d3.event.sourceEvent && d3.event.sourceEvent.type ===
      // "zoom") return; // ignore brush-by-zoom
      // //       // var s = d3.event.selection || x2.range();
      // //       // x.domain(s.map(x2.invert, x2));
      // //       // focus.select(".area").attr("d", area);
      // //       // focus.select(".axis--x").call(xAxis);
      // //       // svg.select(".zoom").call(zoom.transform, d3.zoomIdentity
      // //       //     .scale(width / (s[1] - s[0]))
      // //       //     .translate(-s[0], 0));
      // // // this_.get('mainCanvas').call(this_.get('zoomListener').transform,
      // zoomIdentity);

      this_.get('map').call(this_.get('brushListener'));
      this_.brushed();
      // this_.draw(mainCanvas, false);
    };
    window.onresize(); // let it get span the window
  },
  didDestroyElement() { window.onresize = function() {};},
  databind() {
    let mainCanvas = this.get('mainCanvas');

    let startOfTime = Math.min(...this.get('data').map(e => e.start));
    let endOfTime = Math.max(...this.get('data').map(e => e.end));
    // let timestamps  = this.get('data').map(e => e.start - startOfTime);

    let svgHeight = this.get('height');
    let svgWidth = this.get('width');

    let xScale =
        scaleLinear().domain([ startOfTime, endOfTime ]).range([ 0, svgWidth ]);

    let xScale_brush =
        scaleLinear().domain([ startOfTime, endOfTime ]).range([ 0, svgWidth ]);

    xScale_brush.domain(xScale.domain());

    let cScale = scaleOrdinal(schemeDark2)
                     .domain(this.get('data').map(e => (+e.content.core) % 2))
                     .unknown(scaleImplicit);
    // this.get('data').apply(e => console.log(cScale(e.content.op)));

    let yScale = scaleBand()
                     .domain(this.get('data').map(e => e.tid + e.className))
                     .range([ 0, svgHeight ])
                     .paddingInner(this.get('whiteSpace'));

    this.set('yScale', yScale);
    this.set('xScale', xScale);
    this.set('xScale_brush', xScale_brush);

    // let svg = select(this.$('svg')[0]);

    // svg.attr('height', svgHeight)
    //     .attr('width' , svgWidth)
    //     .on("click", function() {console.log("asd");});

    // svg.selectAll('rect').data(this.get('data'))
    //     .enter()
    //     .append('rect')
    //     .attr('width', (event) => xScale(event.end) - xScale(event.start))
    //     .attr('height', 10)
    //     .attr('pointer-events', 'none')
    //     .attr('style', 'fill:rgb(230,169,0)')
    //     .attr('x', (event) => xScale(event.start))
    //     .attr('y', (event) => yScale(event.tid));

    // var context = mainCanvas.node().getContext('2d');

    var customBase = document.createElement('custom');

    var custom = select(customBase);
    this.set('custom', custom);
    // This is your SVG replacement and the parent of all other elements

    var join = custom.selectAll('custom.rect').data(this.get('data'));

    var enterSel = join.enter()
                       .append('custom')
                       .attr('class', 'rect')
                       .attr('end', 0)
                       .attr('x', (e) => e.start)
                       .attr('y', (e) => e.tid + e.className)
                       .attr('width', 0)
                       .attr('height', 0);

    join.merge(enterSel)
        // .transition()
        .attr('end', (e) => e.end)
        // .attr('width', (e) => xScale(e.end) - xScale(e.start))
        // .attr('fillStyle', function () {return 'rgb(230,169,0)';})
        .attr('fillStyle', (e) => cScale((+e.content.op) % 2))
        .attr('fillStyleHidden', function(d) { return d.rgbid; });

    join.exit()
        // .transition()
        .attr('width', 0)
        .attr('height', 0)
        .remove();

    this.draw(mainCanvas, false);

    var map =
        select('#container')
            .append("svg")
            .attr('width', svgWidth)
            .attr('height', this.get('mapHeight'))
            .attr(
                'style',
                'line-height:0;vertical-align:bottom;font-size:0px;display:block')
    var mapg = map; //.append("g");
    var imageFoo = mapg.append('image');
    imageFoo.attr('href', mainCanvas.node().toDataURL());

    // Style your image here
    imageFoo.attr('width', svgWidth)
        .attr('height', this.get('mapHeight'))
        .attr('position', 'absolute')
        .attr('bottom', '0')
        .attr('left', '0')
        .attr('preserveAspectRatio', 'none')
        .attr(
            'style',
            'line-height:0;vertical-align:bottom;font-size:0px;display:block');

    const this_ = this;

    var brush = brushX()
                    .extent([ [ 0, 0 ], [ svgWidth, this.get('mapHeight') ] ])
                    .on("brush end", () => this_.brushed());

    // this.set('brush', brush);
    this.set('brushListener', brush);

    mapg.call(brush);
    mapg.select(".overlay").attr("fill", "#777").attr("fill-opacity", 0.15)

    mapg.select(".selection").attr("stroke", "#888");

    this.set('map', map);
    this.set('mapImage', imageFoo);
  },
  brushed() {
    if (event && event.sourceEvent && event.sourceEvent.type === "zoom")
      return; // ignore brush-by-zoom
    var s = (event && event.selection) || this.get('xScale_brush').range();
    // xScale.domain(s.map(xScale_brush.invert, xScale_brush))
    // if (d3.event.sourceEvent && d3.event.sourceEvent.type === "zoom") return;
    // // ignore brush-by-zoom var s = d3.event.selection || x2.range();
    // x.domain(s.map(x2.invert, x2));
    // focus.select(".area").attr("d", area);
    // focus.select(".axis--x").call(xAxis);
    // svg.select(".zoom").call(zoom.transform, d3.zoomIdentity
    //     .scale(width / (s[1] - s[0]))
    //     .translate(-s[0], 0));
    this.get('mainCanvas')
        .call(this.get('zoomListener').transform,
              zoomIdentity.scale(this.get('width') / (s[1] - s[0]))
                  .translate(-s[0], 0));
    // this_.draw(mainCanvas, false);
  },
  draw(canvas, hidden) {
    var context = canvas.node().getContext('2d');
    var custom = this.get('custom');

    let svgHeight = this.get('height');
    let svgWidth = this.get('width');

    let yScale = this.get('yScale');
    let xScale = this.get('xScale');

    let height = 5; // yScale.bandwidth();

    context.save();
    context.clearRect(0, 0, svgWidth, svgHeight);
    // if (!hidden) {
    var transform;
    if (event) {
      transform = event.transform;
      if (transform === undefined) {
        transform = zoomTransform(this.get('mainCanvas').node());
      }
    }
    if (transform !== undefined) {
      context.translate(transform.x, 0);
      context.scale(transform.k, 1);
    }
    // }
    // context.clearRect(0, 0, svgWidth, svgHeight); // Clear the canvas.
    const hidden_ = hidden;
    // Draw each individual custom element with their properties.
    var elements = custom.selectAll('custom.rect');
    // Grab all elements you bound data to in the databind() function.
    elements.each(function() { // For each virtual/custom element...
      var node = select(this);
      // This is each individual element in the loop.

      // var e = node.attr('end');
      // console.log(e);
      //     .attr('end', (e) => e.end)
      //     .attr('x', (e) => xScale(e.start))
      //     .attr('y', (e) => yScale(e.tid))
      //     .attr('width', 0)
      //     .attr('height', 0);

      // join.merge(enterSel)
      //     // .transition()
      //     .attr('width', (e) => xScale(e.end) - xScale(e.start))
      //     .attr('height', yScale.bandwidth())

      var x = xScale(node.attr('x'));
      var y = yScale(node.attr('y'));

      var end = node.attr('end');
      var width = 0;
      if (end != 0)
        width = xScale(end) - x;

      context.fillStyle =
          hidden_ ? node.attr('fillStyleHidden') : node.attr('fillStyle');
      // Here you retrieve the color from the individual in-memory node and set
      // the fillStyle for the canvas paint
      context.fillRect(x, y, width, (end != 0) ? height : 0);
      // Here you retrieve the position of the node and apply it to the fillRect
      // context function which will fill and paint the square.
    }); // Loop through each element.

    context.restore();
  }
});
