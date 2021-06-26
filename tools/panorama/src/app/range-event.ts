import {OperationDetails} from './operation-details';


export class EventData {
  private tid: number;
  private content: OperationDetails;

  constructor(
    tid: number,
    content: OperationDetails
  ) {
    this.tid = tid;
    this.content = content;
  }

  public getOp(): number {
    return +this.content.op.op;
  }

  public getClass(): number {
    return Math.floor(this.getOp() / 2);
  }

  public isClassStart(): boolean {
    return (+this.getOp()) % 2 === 0;
  }

  public isSpotEventClass(): boolean {
    return (+this.getOp()) < 0;
  }

  public getGroup(): string {
    return this.content.pipelineId + '::' + this.content.instanceId + '::' + this.getThreadId() + '::' + /* this.content.operator + '::' +*/ this.getClass(); // + '::' + this.getThreadId();
    // this.content.core + '::';// + this.getThreadId(); // + '::' + this.getClass();
  }

  public getOperator(): string {
    return this.content.operator;
  }

  public getCategory(): string {
    return (+this.content.op.op) + '';
  }

  public getThreadId(): number {
    return this.tid;
  }
}

export class Event {
  timestamp: number;
  e: EventData;

  constructor(
    timestamp: number,
    tid: number,
    content: OperationDetails
  ) {
    this.timestamp = timestamp;
    this.e = new EventData(tid, content);
  }

  public getGroup(): string {
    return this.e.getGroup();
  }

  public getCategory(): string {
    return this.e.getCategory();
  }
}

export class RangeEvent {
  /**
   * Start time in milliseconds
   */
  start: number;
  /**
   * End time in milliseconds
   */
  end: number;

  rgbid: string;

  e: EventData;

  constructor(start: number, e: Event) {
    this.start = start;
    this.end = e.timestamp;
    // assert(this.start <= this.end);
    this.e = e.e;
  }

  public getGroup(): string {
    return this.e.getGroup();
  }

  public getCategory(): string {
    return this.e.getCategory();
  }

  public getOperator(): string {
    return this.e.getOperator();
  }
}
