import { TestBed } from '@angular/core/testing';

import { EventTimelineService } from './event-timeline.service';

describe('EventTimelineService', () => {
  let service: EventTimelineService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(EventTimelineService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
