import {Component, ViewChildren, OnInit, QueryList, AfterViewInit} from '@angular/core';
import {TimelineRangeComponent} from '../timeline-range/timeline-range.component';
import {BwProfileComponent} from '../bw-profile/bw-profile.component';
import {TimelineComponent} from '../timeline/timeline.component';

@Component({
  selector: 'app-overview',
  templateUrl: './overview.component.html',
  styleUrls: ['./overview.component.sass']
})
export class OverviewComponent implements OnInit, AfterViewInit {
  @ViewChildren(TimelineComponent) eventTimeline!: QueryList<TimelineComponent>;
  @ViewChildren(TimelineRangeComponent) timeline!: QueryList<TimelineRangeComponent>;
  @ViewChildren(BwProfileComponent) bwprofile!: QueryList<BwProfileComponent>;
  timerange: number[];

  constructor() {
    const filterFromStorage = localStorage.getItem('timerange');
    if (filterFromStorage) {
      const timerange: number[] = JSON.parse(filterFromStorage);
      console.log('Loading timerange from storage: ' + timerange);
    }
  }

  ngOnInit(): void {
  }

  updateSelection(timerange: number[]): void {
    if (!timerange || !this.timerange || this.timerange.some((f, i) => Math.abs(f - timerange[i]) > 1e-9)) {
      console.log('Updating timerange to: ' + timerange);
      localStorage.setItem('timerange', JSON.stringify(timerange));
      this.timerange = timerange;
    }
  }

  ngAfterViewInit(): void {
    const filterFromStorage = localStorage.getItem('timerange');
    if (filterFromStorage) {
      console.log('Loading timerange from storage: ' + filterFromStorage);
      this.updateSelection(JSON.parse(filterFromStorage));
    }
  }
}
