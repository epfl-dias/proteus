package ch.epfl.dias.calcite.adapter.pelago.reporting;


/** Class used for measuring time between two intervals */
public class PelagoTimeInterval {
    private Long startTime;
    private Long endTime  ;

    public void start() {
        startTime = System.nanoTime();
    }

    public void stop() {
        assert(isStarted());
        endTime = System.nanoTime();
    }

    public long getDifferenceNano() {
        assert(isStarted() && endTime != null);
        return endTime-startTime;
    }

    public long getDifferenceMilli() {
        return (getDifferenceNano())/1000000;
    }

    public boolean isStarted() {
        return startTime != null;
    }
}
