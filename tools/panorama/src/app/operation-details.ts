import {Operation} from './operation';

export class OperationDetails {
  op: Operation;

  constructor(
    op: number,
    public core: number,
    public operator: string,
    public coreEnd?: number,
    public pipelineId?: string,
    public instanceId?: number,
  ) {
    this.op = new Operation(op);
  }
}
