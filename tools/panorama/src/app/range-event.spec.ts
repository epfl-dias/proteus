import { RangeEvent, Event } from './range-event';
import {OperationDetails} from './operation-details';

describe('RangeEvent', () => {
  it('should create an instance', () => {
    expect(new RangeEvent(4, new Event(5, 6, new OperationDetails(7, 8, 'asd')))).toBeTruthy();
  });
});
