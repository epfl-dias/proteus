import {Component, EventEmitter, Input, OnInit, Output} from '@angular/core';
import * as d3 from 'd3';
import {Selection} from 'd3-selection';
import {RangeEvent, Event} from '../range-event';
import {schemeDark2} from 'd3-scale-chromatic';
import {Operation} from '../operation';
import {OperationDetails} from '../operation-details';
import {EventTimelineService} from '../event-timeline.service';

@Component({
  selector: 'app-timeline-range',
  templateUrl: './timeline-range.component.html',
  styleUrls: ['./timeline-range.component.sass']
})
export class TimelineRangeComponent implements OnInit {
  @Output() timerange = new EventEmitter<number[]>();
  private privatetimeselection: number[];

  private svg;
  private margin = 50;
  private width = 750 - (this.margin * 2);
  private height = 400 - (this.margin * 2);
  private mapHeight = 50;
  private yScale: d3.ScaleBand<string> = null;
  private xScale: d3.ScaleLinear<number, number, never> = null;
  private xScaleBrush: d3.ScaleLinear<number, number, never> = null;
  private whiteSpace = 0.01;
  private data: RangeEvent[] = null;
  private custom: Selection<HTMLElement, any, null, undefined> = null;
  private mainCanvas: Selection<HTMLElement, any, null, undefined> = null;
  private hiddenCanvas: Selection<HTMLElement, any, null, undefined> = null;
  private zoomListener: d3.ZoomBehavior<Element, unknown>;
  private brush: d3.BrushBehavior<unknown>;
  private map: any;
  private activeBrushOrZoomEvent = false;
  private mapHeightPercent = 0.10;
  private mapImage: any;
  /**
   * Active selection, measured in pixels
   *
   * FIXME: it would be more intuitive to be in `ms` or `sec`
   */
  public selection: number[];

  constructor(private timelineService: EventTimelineService) {
  }

  private createSvg(): void {
    this.svg = d3.select('figure#timeline-range');
    // .append('svg')
    // .attr('width', this.width + (this.margin * 2))
    // .attr('height', this.height + (this.margin * 2))
    // .append('g')
    // .attr('transform', 'translate(' + this.margin + ',' + this.margin + ')');
  }


  private draw(canvas, hidden): void {
    const context = canvas.node().getContext('2d');
    //   // var custom = this.get('custom');
    //
    //   const svgHeight = this.height;
    //   const svgWidth = this.width;
    //
    const height = Math.max(this.yScale.bandwidth(), 5);
    //
    context.save();
    context.clearRect(0, 0, this.width, this.height);

    // Draw each individual custom element with their properties.
    const elements = this.custom.selectAll('custom.rect');
    // Grab all elements you bound data to in the databind() function.
    elements.each((d, i, n) => { // For each virtual/custom element...
      const node = d3.select(n[i]);

      const x = this.xScale(+node.attr('x'));

      if (x > this.width) {
        return;
      }

      const y = this.yScale(node.attr('y'));

      let end = +node.attr('end');
      if (end !== 0) {
        end = this.xScale(end);
      }
      if (end < 0) {
        return;
      }

      const width = end - x;
      context.fillStyle = hidden ? node.attr('fillStyleHidden') : node.attr('fillStyle');
      // Here you retrieve the color from the individual in-memory node and set
      // the fillStyle for the canvas paint
      context.fillRect(x, y, Math.max(width, 5), (end !== 0) ? height : 0);
      // Here you retrieve the position of the node and apply it to the fillRect
      // context function which will fill and paint the square.
    }); // Loop through each element.

    context.restore();
  }

  ngOnInit(): void {
    this.createSvg();
    //
    // this.get('data').forEach((d, index) => {
    //   let ind = index + 1; // 0 is used for empty space
    //   var ret = [ ind & 0xff, (ind & 0xff00) >> 8, (ind & 0xff0000) >> 16 ];
    //   var col = "rgb(" + ret.join(',') + ")";
    //   d.rgbid = col;
    // });

    // const top = d3.select(this).node().getBoundingClientRect().top;
    // let mapHeight =
    //   (this.$(document).height() - top) * this.get('mapHeightPercent');
    // this.set('mapHeight', mapHeight);
    //
    // this.set('height', this.$(document).height() - top - mapHeight);
    // this.set('width', this.$(document).width());
    //
    // let svgHeight = this.get('height');
    // let svgWidth = this.get('width');
    //
    const mainCanvas = this.svg
      .append('xhtml:canvas')
      .classed('mainCanvas', true)
      .attr('top', 0)
      .attr('left', 0)
      .attr('style', 'line-height:0')
      .attr('touch-action', 'none')
      .attr('position', 'absolute')
      .attr('height', this.height)
      .attr('width', this.width);

    this.mainCanvas = mainCanvas;

    const hiddenCanvas =
      this.svg
        .append('xhtml:canvas')
        .classed('hiddenCanvas', true) // select(this.$('canvas')[1])
        .attr('top', 0)
        .attr('left', 0)
        .attr('style', 'line-height:0; display: none;')
        // .attr('style', 'line-height:0')
        .attr('position', 'absolute')
        .attr('height', this.height)
        .attr('width', this.width);

    this.hiddenCanvas = hiddenCanvas;

    mainCanvas.on('mousemove', (event) => {
      // this.draw(hiddenCanvas, true); // Draw the hidden canvas.

      // Get mouse positions from the main canvas.
      const mouseX = event.layerX || event.offsetX;
      const mouseY = event.layerY || event.offsetY;

      const hiddenCtx = hiddenCanvas.node().getContext('2d');

      // Pick the colour from the mouse position.
      const col: number[] = hiddenCtx.getImageData(Math.round(mouseX), Math.round(mouseY), 1, 1).data;
      const col2: number[] = hiddenCtx.getImageData(Math.round(mouseX - 1), Math.round(mouseY - 1), 1, 1).data;
      const col3: number[] = hiddenCtx.getImageData(Math.round(mouseX + 1), Math.round(mouseY + 1), 1, 1).data;

      // Then stringify the values in a way our map-object can read it.

      const colKey = (col.every((f, i) => f === col2[i]) && col.every((f, i) => f === col3[i]))
        // tslint:disable-next-line:no-bitwise
        ? ((+col[0]) | (+col[1] << 8) | (+col[2] << 16))
        : 0;

      // Get the data from our map!
      if (colKey !== 0) {
        const nodeData = this.data[colKey - 1];
        d3.select('#tooltip')
          .style('opacity', 0.8)
          .style('top', event.pageY + 5 + 'px')
          .style('left', event.pageX + 5 + 'px')
          .style('height', 0)
          .style('width', 0)
          .html(JSON.stringify(nodeData, (key, value) => {
            if (value instanceof Operation) {
              return '' + Operation.opNames[value.op];
            }
            return value;
          }, 2));
      } else {
        d3.select('#tooltip')
          .style('opacity', 0)
          .style('height', 0)
          .style('width', 0)
          .html('');
      }
    });

    // this.set('mainCanvas', mainCanvas);
    // this.databind();
    // console.log(d3.select('#container').node().getBoundingClientRect().top);
    //
    window.onresize = () => this.resize();


    Promise.all([this.timelineService.getTimeline()
      .toPromise()
      .then(data => {
        return (data.find(d => Operation.opNames[d.e.getOp()] === 'CUPTI_GET_START_TIMESTAMP') || data[0]).start;
      }), d3.csv('/assets/cupti.csv')
    ]).then(d => {
        const startOfTime = d[0];
        const data = d[1];
        return data.map((item) => {
          const details = new OperationDetails(-1, -1, 'GPU Kernel: ' + item.kernel_name);
          const event = new Event((+item.end / (1000 * 1000)) + startOfTime, -1, details);
          return new RangeEvent((+item.start / (1000 * 1000)) + startOfTime, event);
        });
      }
    )
      .then(data => {
        return data.map((e, index) => {
          e.start = Math.floor(e.start * 100000) / 100000;
          e.end = Math.floor(e.end * 100000) / 100000;
          const ind = index + 1; // 0 is used for empty space
          // tslint:disable-next-line:no-bitwise
          const ret = [ind & 0xff, (ind & 0xff00) >> 8, (ind & 0xff0000) >> 16];
          e.rgbid = 'rgb(' + ret.join(',') + ')';
          return e;
        });
      })
      .then(data => {
        const startOfTime = 0; // data.map(e => e.start).reduce((e1, e2) => Math.min(e1, e2));
        const endOfTime = (data.length !== 0) ? data.map(e => e.end).reduce((e1, e2) => Math.max(e1, e2)) : startOfTime;

        this.xScale =
          d3.scaleLinear().domain([startOfTime, endOfTime]).range([0, this.width]);

        this.xScaleBrush =
          d3.scaleLinear().domain([startOfTime, endOfTime]).range([0, this.width]);
        this.selection = this.xScaleBrush.range();


        this.xScaleBrush.domain(this.xScale.domain());

        const cScale = d3.scaleOrdinal(schemeDark2)
          .domain(data.map(e => e.getCategory()))
          .unknown(d3.scaleImplicit);
        // this.get('data').apply(e => console.log(cScale(e.content.op)));

        const ycalc = (e: RangeEvent) => {
          const t = e.getOperator();
          return t.slice(t.lastIndexOf('('));
        };

        this.yScale = d3.scaleBand()
          .domain(data.map(ycalc).sort().filter((value, index, self) => index === 0 || self[index - 1] !== value))
          .range([0, this.height])
          .paddingInner(this.whiteSpace);

        const customBase = document.createElement('custom');

        this.custom = d3.select(customBase);
        // this.set('custom', custom);
        // This is your SVG replacement and the parent of all other elements

        const join = this.custom.selectAll('custom.rect').data(data);

        if (data) {
          const enterSel = join.enter()
              .append('custom')
              .attr('class', 'rect')
              .attr('end', 0)
              .attr('x', (e) => e.start)
              .attr('y', ycalc)
              .attr('width', 5)
              .attr('height', 5);

          join.merge(enterSel)
              .attr('end', (e) => e.end)
              .attr('fillStyle', (e) => cScale(e.getCategory()))
              .attr('fillStyleHidden', (d) => d.rgbid);

          join.exit()
              .attr('width', 0)
              .attr('height', 0)
              .remove();
        }

        this.data = data;
        this.draw(mainCanvas, false);

        this.map =
          d3.select('figure#timeline-range')
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.mapHeight)
            .attr(
              'style',
              'line-height:0;vertical-align:bottom;font-size:0px;display:block');

        const imageFoo = this.map.append('image');
        imageFoo.attr('href', mainCanvas.node().toDataURL());
        this.mapImage = imageFoo;

        imageFoo.attr('width', this.width)
          .attr('height', this.mapHeight)
          .attr('position', 'absolute')
          .attr('bottom', '0')
          .attr('left', '0')
          .attr('preserveAspectRatio', 'none')
          .attr(
            'style',
            'line-height:0;vertical-align:bottom;font-size:0px;display:block');

        this.brush = d3.brushX()
          .extent([[0, 0], [this.width, this.mapHeight]])
          .on('end', (event) => this.brushed(event));

        this.zoomListener =
          d3.zoom()
            .translateExtent([[0, 0], [this.width, 0]])
            .scaleExtent([1, Infinity])
            .on('zoom', (event) => this.zoomed(event))
            .touchable(true);

        this.zoomListener(mainCanvas);

        this.map.call(this.brush);
        this.map.select('.overlay').attr('fill', '#777').attr('fill-opacity', 0.15);

        this.map.select('.selection').attr('stroke', '#888');
        this.resize();
      })
      .then(() => {
        console.log('Setting time selection from ngOnInit to ' + this.privatetimeselection);
        this.select(this.privatetimeselection);
      });
  }

  private zoomed(event: any): void {
    if (this.activeBrushOrZoomEvent) {
      this.timerange.emit(this.selection.map(this.xScaleBrush.invert).map(x => x / 1000));
      return; // ignore zoom-by-brush
    }

    this.activeBrushOrZoomEvent = true;
    const t = event.transform; // || this.xScaleBrush.domain();
    console.log(t);
    this.xScale.domain(t.rescaleX(this.xScaleBrush).domain());

    this.map.call(this.brush.move,
      this.xScale.range().map(t.invertX, t));

    this.draw(this.mainCanvas, false);
    this.draw(this.hiddenCanvas, true);
    this.activeBrushOrZoomEvent = false;
  }

  private brushed(event: { selection: number[] }): void {
    // selection is measured in pixels and corresponds to xScale.range()
    const s: number[] = event.selection || this.selection || this.xScale.range();
    if (this.selection !== undefined && s.every((f, i) => Math.abs(f - this.selection[i]) < 1e-9)) {
      return;
    }

    this.selection = s;
    console.log('brushed-range: ' + s);
    if (this.activeBrushOrZoomEvent) {
      this.timerange.emit(this.selection.map(this.xScaleBrush.invert).map(x => x / 1000));
      return; // ignore brush-by-zoom
    }

    this.activeBrushOrZoomEvent = true;
    this.xScale.domain(s.map(this.xScaleBrush.invert, this.xScaleBrush));

    this.mainCanvas.call(this.zoomListener.transform,
      d3.zoomIdentity.scale(this.width / (s[1] - s[0]))
        .translate(-s[0], 0));

    this.draw(this.mainCanvas, false);
    this.draw(this.hiddenCanvas, true);
    this.activeBrushOrZoomEvent = false;
  }

  public select(timerange: number[]): void {
    if (timerange === undefined || !this.xScaleBrush) {
      return;
    }

    timerange = timerange.map(x => x * 1000).map(this.xScaleBrush);

    this.xScale.domain(timerange);
    this.map.call(this.brush.move, timerange);
  }

  private resize(/*event*/): void {
    const fig = d3.select('figure#timeline-range').attr('style', 'margin-top: 0px; margin-bottom: 0px');

    // Align canvas with Vega axis
    fig.attr('style', 'margin-top: 0px; margin-bottom: 0px; margin-left: 58px');
    // @ts-ignore
    const bb = fig.node().getBoundingClientRect();
    const top = +bb.top;
    const bbheight = +window.innerHeight / 8 - 16 /* body top+bottom padding */;
    const mapHeight =
      (bbheight - top) * this.mapHeightPercent;
    this.mapHeight = mapHeight;

    const oldHeight = this.height;
    const oldWidth = this.width;

    this.height = bbheight - top - mapHeight;
    this.width = 1500; // +bb.width;

    const s = this.selection;
    const sold = s.map(this.xScaleBrush.invert, this.xScaleBrush);

    this.yScale.range([0, this.height]);
    this.xScale.range([0, this.width]);
    this.xScaleBrush.range([0, this.width]);
    this.zoomListener.translateExtent([[0, 0], [this.width, 0]]);
    this.selection = sold.map(this.xScaleBrush);

    this.mainCanvas.attr('width', this.width).attr('height', this.height);
    this.hiddenCanvas.attr('width', this.width).attr('height', this.height);

    this.map.attr('width', this.width).attr('height', mapHeight);
    this.mapImage.attr('width', this.width).attr('height', mapHeight);

    if (this.height > oldHeight || this.width > oldWidth) {
      // TODO: redraw minimap, as its resolution may have degraded

      // quality has been improved in at least one dimension update map
      // this.mapImage.attr('href', this.mainCanvas.node().toDataURL());
    }

    this.brush.extent([
      [0, 0], [this.width, this.mapHeight]
    ]);
    this.map.call(this.brush);

    // @ts-ignore
    this.brush.move(this.map, s.map((y: number) => y / (oldWidth / this.width)));
    this.brushed(this);

    this.draw(this.mainCanvas, false);
    this.draw(this.hiddenCanvas, true);
  }

  @Input()
  set timeselection(e: number[]) {
    console.log('Timeselection set to: ' + e);
    this.privatetimeselection = e;
    this.select(e);
  }
}
