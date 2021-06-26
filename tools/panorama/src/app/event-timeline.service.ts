import {Injectable} from '@angular/core';
import {from, Observable} from 'rxjs';
import {Event, RangeEvent} from './range-event';
import * as d3 from 'd3';
import {Operation} from './operation';
import {OperationDetails} from './operation-details';
import {publishReplay, refCount} from 'rxjs/operators';
import {DSVRowArray} from 'd3';

function memoize(): (target: any, propertyKey: string, descriptor: PropertyDescriptor) => void {
  return (
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) => {
    const method = descriptor.value; // references the method being decorated
    const cacheMember = propertyKey + 'CacheMember';

    // the Observable function
    if (!descriptor.value) {
      throw new Error('use MemoizeDecorator only on services methods');
    }

    descriptor.value = (...args) => {
      if (!target[cacheMember]) {
        const returnedObservable = method.apply(this, args);
        if (!(returnedObservable instanceof Observable)) {
          throw new Error(
            `method decorated with Memoized Decorator must return Observable`
          );
        }

        target[cacheMember] = returnedObservable.pipe(
          publishReplay(),
          refCount()
        );
      }

      return target[cacheMember];
    };
  };
}

@Injectable({
  providedIn: 'root'
})
export class EventTimelineService {

  constructor() {
  }

  // @memoize()
  private getRawData(): Observable<DSVRowArray<string>> {
    return from(d3.csv('/assets/timeline-oplegend.csv')
      .then((d) => {
        d.forEach((e) => {
          Operation.opNames[+e.value] = e.op;
        });
      }).then(() => d3.csv('/assets/timeline.csv')));
  }

  private getRawRangeData(): Observable<DSVRowArray<string>> {
    return from(d3.csv('/assets/timeline-ranges-oplegend.csv')
      .then((d) => {
        d.forEach((e) => {
          Operation.opNames[+e.value] = e.op;
        });
      }).then(() => d3.csv('/assets/timeline-ranges.csv')));
  }

  // @memoize()
  getKFrequency(): Observable<number> {
    return from(this.getRawRangeData().toPromise().then(data => {
      const timestamps = data.filter(d => Operation.opNames[+d.op] === 'LOGGER_TIMESTAMP').sort((f1, f2) => (+f1.timestamp_start - +f2.timestamp_start));
      const firstTick = +timestamps[0].timestamp_start;
      const lastTick = +timestamps[timestamps.length - 1].timestamp_start;
      const firstStamp = parseInt(timestamps[0].operator, 16); // ns
      const lastStamp = parseInt(timestamps[timestamps.length - 1].operator, 16); // ns
      console.log(((lastStamp - firstStamp) / (1000 * 1000)));
      const kfreq = (lastTick - firstTick) / ((lastStamp - firstStamp) / (1000 * 1000));
      console.log('Computed kfreq: ' + kfreq);
      return kfreq;
    }));
  }

  // @memoize()
  getTimeline(): Observable<RangeEvent[]> {
    return from(Promise.all([Promise.all([this.getRawData().toPromise(), this.getKFrequency().toPromise()]).then(tmp => {
      const data = tmp[0];
      const kfreq = tmp[1];


      return data.sort((f1, f2) => (+f1.timestamp - +f2.timestamp)).map(d => new Event(
        ((+d.timestamp) / kfreq),
        +d.thread_id,
        new OperationDetails(
          +d.op,
          +d.coreid,
          d.operator
        )));
    }).then(data => {
        // // assert(data.length > 0);
        const out: RangeEvent[] = [];
        const d: Record<number, Record<number, Event>> = {};
        data.forEach((item) => {
          if (item.e.isSpotEventClass()) {
            //     item.start = item.timestamp;
            //     item.end = item.timestamp + 0.0001;
            //     delete item.timestamp;
            //
            //     // out.push(data[i])
          } else {
            // Positive classNames represent time ranges
            // if (+(item.e.className) / 2 === 6 / 2) {return; }
            // if (+(item.e.className) / 2 === 7 / 2) {return; }

            if (!item.e.isClassStart()) {
              const x = d[item.e.getThreadId()][item.e.getClass()];
              if (x === undefined) {
                console.log(Operation.opNames[item.e.getClass()]);
                return;
              }
              // assert(x.timestamp < item.timestamp);
              out.push(new RangeEvent(x.timestamp, item));
              delete d[item.e.getThreadId()][item.e.getClass()];
            } else {
              if (!(item.e.getThreadId() in d)) {
                d[item.e.getThreadId()] = {};
              }
              d[item.e.getThreadId()][item.e.getClass()] = item;
            }
          }
        });
        console.log(out.length);
        return out;
      }
    ), Promise.all([this.getRawRangeData().toPromise(), this.getKFrequency().toPromise()]).then(d => {
      const ranges = d[0];
      const kfreq = d[1];

      return ranges.map(item => {
        return new RangeEvent(
          (+item.timestamp_start) / kfreq,
          new Event(
            (+item.timestamp_end) / kfreq,
            +item.thread_id,
            new OperationDetails(
              +item.op,
              +item.coreid_start,
              item.operator,
              +item.coreid_end,
              item.pipeline_id,
              +item.instance_id
            )
          )
        );
      });
    })]).then(d => {
      return d[0].concat(d[1]);
    }).then(data => {
      const t = data.map(e => e.start).reduce((e1, e2) => Math.min(e1, e2));
      console.log('minimum time: ' + t);
      return data.map((e, index) => {
        e.start -= t;
        e.end -= t;
        return e;
      });
    })
  );
  }
}
