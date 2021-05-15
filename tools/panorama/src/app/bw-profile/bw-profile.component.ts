import {Component, OnInit} from '@angular/core';
import * as d3 from 'd3';
import VisualizationSpec, {default as vegaEmbed} from 'vega-embed';
import {JoinAggregateTransformNode} from 'vega-lite/build/src/compile/data/joinaggregate';
import {JoinAggregateFieldDef, JoinAggregateTransform} from 'vega-lite/src/transform';
import {VgJoinAggregateTransform} from 'vega-lite/src/vega.schema';

class TimeAdjustment {
  start: number;
  freq: number;

  constructor(start: number, end: number, freq: number) {
    this.start = start;
    this.freq = freq;
  }
}

@Component({
  selector: 'app-bw-profile',
  templateUrl: './bw-profile.component.html',
  styleUrls: ['./bw-profile.component.sass']
})
export class BwProfileComponent implements OnInit {

  // var margin = {top: 20, right: 20, bottom: 40, left: 40}
  // width = 500
  // height = 500,
  // gap = 0,
  // ease = 'cubic-in-out';
  // var svg, duration = 500;


  constructor() {
  }

  private svg;

  static adjustTimestamps<T>(v: (T & { timestamp: number })[], adjust: TimeAdjustment): (T & { timestamp: number })[] {
    return v.map(e => {
      e.timestamp = ((e.timestamp - adjust.start) / adjust.freq);  /* sec */
      return e;
    });
  }

  private createSvg(): void {
    this.svg = d3.select('div#bw-profile').node();
    console.log(this.svg);
    // .append('svg')
    // .attr('width', this.width + (this.margin * 2))
    // .attr('height', this.height + (this.margin * 2))
    // .append('g')
    // .attr('transform', 'translate(' + this.margin + ',' + this.margin + ')');
  }

  ngOnInit(): void {
    this.createSvg();

    const buffRaw = d3.csv('/assets/ib-buffs.csv').then(
      d => d.map(e => ({timestamp: +e.timestamp, buffs: +e.buffs, handler: e.handler})).sort()
    );

    const params = buffRaw.then(d => {
      const start = d[0].timestamp;
      const end = d[d.length - 1].timestamp;
      console.log(end - start);
      const freq = (end - start) / 69.0; // ticks per second
      return new TimeAdjustment(start, end, freq);
    });

    const buffs = Promise.all([buffRaw, params]).then(dall => {
      const d = dall[0];
      return BwProfileComponent.adjustTimestamps(d, dall[1]);
    });

    const slack = Promise.all([d3.csv('/assets/slack-log.csv').then(
      d => d.map(e => ({timestamp: +e.timestamp, avail_slack: +e.avail_slack, handler: e.handler})).sort()
    ), params]).then(dall => {
      const d = dall[0];
      return BwProfileComponent.adjustTimestamps(d, dall[1]);
    });

    const ibLog = Promise.all([d3.csv('/assets/ib-log.csv').then(d => d.map(e => ({
      timestamp: +e.timestamp,
      MB: (+e.bytes / (1024.0 * 1024))
    })).sort()), params]).then(dall => BwProfileComponent.adjustTimestamps(dall[0], dall[1]));

    const bw = ibLog.then(val => {
      let i = 0;
      const data = [];
      let size = 0.0;
      let time = 0.0;
      const bwGBps = 25.0; // GBps
      const step = 25.0 / (1000 * 1000); // usec
      const mbperstep = (bwGBps * step) * 1024;
      console.log(mbperstep);
      console.log('--------------------');
      while (i < val.length || size > 0) {
        while (i < val.length && val[i].timestamp < time) {
          size += val[i].MB;
          ++i;
        }

        const consume = Math.min(size, mbperstep);

        data.push({
          timestamp: time,
          MB: consume / step,
        });
        size -= consume;
        time += step;
      }

      return data;
    });

    Promise.all([bw, buffs, slack, ibLog]).then(v => {
      const bwReady = v[0];
      const buffsReady = v[1];
      const slackReady = v[2];
      const ibCntReady = v[3];

      console.log(v);
      vegaEmbed('#vega-lite', {
          $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
          // padding: 5,

          vconcat: [{
            data: {
              values: bwReady
            },
            width: 1500,
            height: 400,
            params: [{
              name: 'brush',
              select: {
                type: 'interval',
                encodings: ['x'],
              },
              bind: 'scales'
            }],
            mark: 'bar',
            encoding: {
              x: {
                field: 'timestamp',
                type: 'quantitative',
                bin: {
                  maxbins: 100,
                  extent: {
                    param: 'brush'
                  },
                },
                axis: {
                  labelAngle: -45,
                  labelOverlap: false,
                },
                title: 'time (s)',
              },
              y: {
                aggregate: 'average', // sum',
                field: 'MB'
              },
            },
          }, {
            data: {
              values: buffsReady
            },
            width: 1500,
            height: 100,
            params: [{
              name: 'brush',
              select: {
                type: 'interval',
                encodings: ['x'],
              },
              bind: 'scales'
            }],
            mark: {
              type: 'line',
              interpolate: 'step-after'
            },
            encoding: {
              x: {
                field: 'timestamp',
                type: 'quantitative',
                // bin: {
                //   maxbins: 100,
                //   extent: {
                //     // @ts-ignore
                //     selection: 'brush',
                //   },
                // },
                axis: {
                  // format: '%Y-%m-%d %H:%m:%S.%L',
                  labelAngle: -45,
                  labelOverlap: false,
                },
                title: 'time (s)'
              },
              y: {
                field: 'buffs',
                type: 'quantitative',
                axis: {
                  labelOverlap: false,
                },
              },
              color: {
                field: 'handler'
              },
            },
          }, {
            data: {
              values: slackReady
            },
            width: 1500,
            height: 300,
            params: [{
              name: 'brush',
              select: {
                type: 'interval',
                encodings: ['x'],
              },
              bind: 'scales'
            }],
            mark: {
              type: 'line',
              interpolate: 'step-after'
            },
            transform: [
              {
                joinaggregate: [{
                  field: 'avail_slack',
                  op: 'max',
                  as: 'capacity_slack'
                } as JoinAggregateFieldDef],
                groupby: ['handler']
              },
              {
                calculate: 'datum.avail_slack / max(datum.capacity_slack, 1)',
                as: 'avail_slack_percent'
              }
            ],
            encoding: {
              x: {
                field: 'timestamp',
                type: 'quantitative',
                // bin: {
                //   maxbins: 100,
                //   extent: {
                //     // @ts-ignore
                //     selection: 'brush',
                //   },
                // },
                axis: {
                  // format: '%Y-%m-%d %H:%m:%S.%L',
                  labelAngle: -45,
                  labelOverlap: false,
                },
                title: 'time (s)',
              },
              y: {
                field: 'avail_slack_percent',
                type: 'quantitative',
                axis: {
                  labelOverlap: false,
                },
              },
              color: {
                condition: {
                  param: 'hover',
                  field: 'handler',
                  type: 'nominal',
                  legend: null
                },
                value: 'grey',
              },
              opacity: {
                condition: {
                  param: 'hover',
                  value: 1
                },
                value: 0.1
              }
            },
            layer: [{
              description: 'transparent layer to make it easier to trigger selection',
              params: [{
                name: 'hover',
                select: {
                  type: 'point',
                  fields: ['avail_slack_percent'],
                  on: 'mouseover'
                }
              }],
              mark: {
                type: 'line',
                stroke: 'transparent',
                interpolate: 'step-after'
              }
            }, {
              params: [{
                name: 'brush',
                select: {
                  type: 'interval',
                  encodings: ['x'],
                },
                bind: 'scales'
              }],
              mark: {
                type: 'line',
                interpolate: 'step-after'
              },
              // }, {
              //   encoding: {
              //     x: {aggregate: 'max', field: 'timestamp'},
              //     y: {aggregate: {argmax: 'timestamp'}, field: 'avail_slack'}
              //   },
              //   layer: [{
              //     mark: {type: 'circle'}
              //   }, {
              //     mark: {type: 'text', align: 'left', dx: 4},
              //     encoding: {text: {field: 'avail_slack', type: 'nominal'}}
              //   }]
            }]
          }, {
            data: {
              values: ibCntReady
            },
            width: 1500,
            height: 100,
            selection: {
              brush: {
                type: 'interval',
                encodings: ['x'],
                bind: 'scales',
              },
            },
            transform: [
              {calculate: 'datum.MB >= 1', as: 'big'}
            ],
            mark: 'bar',
            encoding: {
              x: {
                field: 'timestamp',
                type: 'quantitative',
                bin: {
                  maxbins: 100,
                  extent: {
                    // @ts-ignore
                    selection: 'brush',
                  },
                },
                axis: {
                  labelAngle: -45,
                  labelOverlap: false,
                },
                title: 'time (s)',
              },
              y: {
                aggregate: 'count',
              },
              color: {
                field: 'big'
              },
            },
          }],
          config: {view: {stroke: null}}
        }
      );
    });
  }
}
