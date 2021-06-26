import {Component, EventEmitter, Input, OnInit, Output} from '@angular/core';
import * as d3 from 'd3';
import {default as vegaEmbed} from 'vega-embed';
import {JoinAggregateTransformNode} from 'vega-lite/build/src/compile/data/joinaggregate';
import {JoinAggregateFieldDef, JoinAggregateTransform} from 'vega-lite/src/transform';
import {VgJoinAggregateTransform} from 'vega-lite/src/vega.schema';
import {timestamp} from 'rxjs/operators';
import {VisualizationSpec, Result} from 'vega-embed';
import {EventTimelineService} from '../event-timeline.service';
import {Operation} from '../operation';

class TimeAdjustment {
  constructor(public start: number, public end: number, public freq: number, public startOffsetInTime: number) {
  }
}

@Component({
  selector: 'app-bw-profile',
  templateUrl: './bw-profile.component.html',
  styleUrls: ['./bw-profile.component.sass']
})
export class BwProfileComponent implements OnInit {
  @Output() timerange = new EventEmitter<number[]>();
  private privatetimeselection: number[];

  // var margin = {top: 20, right: 20, bottom: 40, left: 40}
  // width = 500
  // height = 500,
  // gap = 0,
  // ease = 'cubic-in-out';
  // var svg, duration = 500;
  private plots: Promise<Result>;

  constructor(private timelineService: EventTimelineService) {
  }

  static adjustTimestamps<T>(v: (T & { timestamp: number })[], adjust: TimeAdjustment): (T & { timestamp: number })[] {
    return v.map(e => {
      e.timestamp = ((e.timestamp - adjust.start) / adjust.freq) + adjust.startOffsetInTime;  /* sec */
      return e;
    });
  }

  ngOnInit(): void {
    const buffRaw = d3.csv('/assets/ib-buffs.csv').then(
      d => d.map(e => ({timestamp: +e.timestamp, buffs: +e.buffs, handler: e.handler})).sort()
    );

    const params = Promise.all([buffRaw, this.timelineService.getTimeline()
      .toPromise()
      .then(data => {
        return data.find(d => Operation.opNames[d.e.getOp()] === 'IB_BUFFS_GET_START_TIMESTAMP').start /  /* to convert to sec */ 1000.0;
      }), this.timelineService.getKFrequency()
      .toPromise()]).then(x => {
      console.log(x[1]);
      const d = x[0];
      const start = d[0].timestamp;
      const end = d[d.length - 1].timestamp;
      console.log(end - start);
      const freq = x[2] * 1000;
      console.log('offset: ' + x[1] + ' freq: ' + freq);
      return new TimeAdjustment(start, end, freq, x[1]);
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
      handler: +e.handler,
      MB: (+e.bytes / (1024.0 * 1024))
    })).sort()), params]).then(dall => BwProfileComponent.adjustTimestamps(dall[0], dall[1]));

    const bw = ibLog.then(val => {
      let i = 0;
      const data = [];
      const size: { [id: number]: number; } = {};
      let time = val[0].timestamp;
      const bwGBps = 12.5; // GBps
      const step = 25 * 100 / (1000 * 1000); // usec
      const mbperstep = (bwGBps * step) * 1024;
      console.log(mbperstep);
      console.log('--------------------');
      while (i < val.length) {
        while (i < val.length && val[i].timestamp < time) {
          if (!(val[i].handler in size)) {
            size[val[i].handler] = 0;
          }
          size[val[i].handler] += val[i].MB;
          ++i;
        }

        let anyPos = false;
        for (const key in size) {
          const consume = Math.min(size[key], mbperstep);
          data.push({
            timestamp: time,
            MB: consume / step,
            handler: key,
          });
          size[key] -= consume;
          anyPos = anyPos || (size[key] > 0);
        }
        if (!anyPos && i >= val.length) {
          break;
        }
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
      this.plots = vegaEmbed('#vega-lite', {
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
              color: {
                field: 'handler',
                type: 'ordinal',
              }
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
            mark: 'image',
            url: {value: 'https://vega.github.io/images/idl-logo.png'},
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
              enter: {
                url: {value: 'https://vega.github.io/images/idl-logo.png'},
              }
            },
          }],
          resolve: {scale: {x: 'shared'}},
          config: {view: {stroke: null}}
        }
      ).then(plot => {
        this.timerange.emit(plot.view.signal('brush').timestamp);
        plot.view.addSignalListener('brush', (signal, e) => {
          console.log(signal + ' new val in listener: ' + JSON.stringify(e));
          this.timerange.emit(e.timestamp);
        });

        return plot;
      }).then(plot => {
        this.select(this.privatetimeselection);
        return plot;
      });
    });
  }

  private eqArray(a: number[], b: number[]): boolean {
    if (!a || !b || a.length !== b.length) {
      return false;
    }
    return a.every((f, i) => Math.abs(f - b[i]) < 1e-9);
  }

  select(timerange: number[]): void {
    this.plots.then(plot => {
      if (!this.eqArray(plot.view.signal('brush').timestamp, timerange)) {
        console.log('new selection: ' + plot.view.signal('brush').timestamp + ' => ' + timerange);
        plot.view.signal('brush_timestamp', timerange).runAsync();
      }
    });
  }

  @Input()
  set timeselection(e: number[]) {
    console.log('Timeselection set to: ' + e);
    this.privatetimeselection = e;
    this.select(e);
  }
}
