import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TimelineRangeComponent } from './timeline-range.component';

describe('TimelineRangeComponent', () => {
  let component: TimelineRangeComponent;
  let fixture: ComponentFixture<TimelineRangeComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ TimelineRangeComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(TimelineRangeComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
